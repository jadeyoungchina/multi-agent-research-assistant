import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.domain.research import Citation, ReportFinding, ResearchReport
from app.domain.runs import ResearchRun
from app.storage.database import Database
from app.storage.runs import RunRepository


@pytest.fixture
def database(tmp_path: Path) -> Database:
    value = Database(tmp_path / "app.db")
    value.initialize()
    return value


def _state(run_id: str, next_stage: str = "planner") -> dict:
    return {
        "run_id": run_id,
        "next_stage": next_stage,
        "provider_metrics": [],
        "note": "保留原文",
    }


def _report(evidence_sufficient: bool = True) -> ResearchReport:
    return ResearchReport(
        title="Result",
        summary="Summary",
        findings=[
            ReportFinding(
                heading="Finding",
                narrative="Supported claim",
                evidence_ids=["chunk-1"],
            )
        ],
        limitations=[],
        citations=[
            Citation(
                evidence_id="chunk-1",
                filename="paper.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="Evidence",
            )
        ],
        markdown="# Result",
        evidence_sufficient=evidence_sufficient,
    )


def test_run_state_and_events_round_trip(database: Database) -> None:
    repository = RunRepository(database)
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    state = _state(run.id)

    repository.create(run, state)
    started = repository.start_stage(run.id, state, "planner")
    completed = {**state, "next_stage": "retriever", "last_completed_stage": "planner"}
    event = repository.complete_stage(
        run.id, completed, "retriever", "planner", 0, [], {"attempt": 1}
    )

    assert repository.get(run.id) == run.model_copy(
        update={
            "status": "running",
            "current_stage": "retriever",
            "last_completed_stage": "planner",
            "updated_at": repository.get(run.id).updated_at,
        }
    )
    assert repository.get_state(run.id) == completed
    assert started.sequence == 2
    assert event.sequence == 3
    assert [item.sequence for item in repository.list_events(run.id, after_sequence=0)] == [1, 2, 3]
    assert [item.sequence for item in repository.list_events(run.id, after_sequence=1)] == [2, 3]
    with database.connect() as connection:
        stored_state = connection.execute(
            "SELECT state_json FROM research_runs WHERE id = ?", (run.id,)
        ).fetchone()[0]
    assert "保留原文" in stored_state


def test_complete_stage_requires_current_stage_to_match_checkpoint(database: Database) -> None:
    repository = RunRepository(database)
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    state = _state(run.id)
    repository.create(run, state)

    with pytest.raises(ValueError, match="next_stage"):
        repository.complete_stage(
            run.id,
            {**state, "next_stage": "retriever"},
            "writer",
            "planner",
            0,
            [],
            {},
        )

    assert repository.get(run.id) == run
    assert len(repository.list_events(run.id, 0)) == 1


def test_final_stage_persists_report_aggregates_and_one_finished_event(
    database: Database,
) -> None:
    repository = RunRepository(database)
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    initial = _state(run.id, "citation_validator")
    repository.create(run, initial)
    repository.start_stage(run.id, initial, "citation_validator")
    metrics = [
        {
            "provider": "fake",
            "model": "one",
            "latency_ms": 10,
            "retries": 2,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
        {
            "provider": "fake",
            "model": "two",
            "latency_ms": 20,
            "retries": 1,
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        },
    ]
    final_state = {
        **initial,
        "next_stage": "completed",
        "last_completed_stage": "citation_validator",
        "provider_metrics": metrics,
    }
    report = _report()

    repository.complete_stage(
        run.id,
        final_state,
        "completed",
        "citation_validator",
        2,
        metrics,
        {"attempt": 1},
        report,
    )

    stored = repository.get(run.id)
    assert stored is not None
    assert stored.status == "completed"
    assert stored.current_stage == "completed"
    assert stored.last_completed_stage == "citation_validator"
    assert stored.iteration == 2
    assert (stored.prompt_tokens, stored.completion_tokens, stored.total_tokens) == (15, 10, 25)
    assert stored.model_call_count == 2
    assert stored.retry_count == 3
    assert stored.evidence_sufficient is True
    assert stored.completed_at is not None
    assert repository.get_report(run.id) == report
    events = repository.list_events(run.id, 0)
    finished = [event for event in events if event.event_type == "finished"]
    assert len(finished) == 1
    assert finished[0].payload == {"status": "completed", "evidence_sufficient": True}


def test_start_stage_rolls_back_snapshot_and_status_when_event_insert_fails(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = RunRepository(database)
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    state = _state(run.id)
    repository.create(run, state)

    def fail_event(*args, **kwargs):
        raise RuntimeError("injected event failure")

    monkeypatch.setattr(repository, "_insert_event", fail_event)

    with pytest.raises(RuntimeError, match="injected event failure"):
        repository.start_stage(run.id, {**state, "scratch": "new"}, "planner")

    assert repository.get(run.id) == run
    assert repository.get_state(run.id) == state
    assert len(repository.list_events(run.id, 0)) == 1


def test_complete_stage_rolls_back_checkpoint_report_and_events_on_late_failure(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = RunRepository(database)
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    initial = _state(run.id, "citation_validator")
    repository.create(run, initial)
    repository.start_stage(run.id, initial, "citation_validator")
    before = repository.get(run.id)
    before_events = repository.list_events(run.id, 0)
    original_insert = repository._insert_event

    def fail_finished(connection, run_id, stage, event_type, payload):
        if event_type == "finished":
            raise RuntimeError("injected finished-event failure")
        return original_insert(connection, run_id, stage, event_type, payload)

    monkeypatch.setattr(repository, "_insert_event", fail_finished)
    final_state = {
        **initial,
        "next_stage": "completed",
        "last_completed_stage": "citation_validator",
    }

    with pytest.raises(RuntimeError, match="injected finished-event failure"):
        repository.complete_stage(
            run.id,
            final_state,
            "completed",
            "citation_validator",
            0,
            [],
            {},
            _report(),
        )

    assert repository.get(run.id) == before
    assert repository.get_state(run.id) == initial
    assert repository.get_report(run.id) is None
    assert repository.list_events(run.id, 0) == before_events


def test_fail_stage_preserves_start_snapshot_and_rolls_back_failure_on_event_error(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = RunRepository(database)
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    start_snapshot = _state(run.id)
    repository.create(run, start_snapshot)
    repository.start_stage(run.id, start_snapshot, "planner")
    before = repository.get(run.id)
    before_events = repository.list_events(run.id, 0)

    def fail_event(*args, **kwargs):
        raise RuntimeError("injected event failure")

    monkeypatch.setattr(repository, "_insert_event", fail_event)

    with pytest.raises(RuntimeError, match="injected event failure"):
        repository.fail_stage(
            run.id,
            {**start_snapshot, "partial_result": "must not persist"},
            "planner",
            "provider_error",
            "failed",
        )

    assert repository.get(run.id) == before
    assert repository.get_state(run.id) == start_snapshot
    assert repository.list_events(run.id, 0) == before_events


def test_fail_stage_marks_run_failed_without_overwriting_start_snapshot(database: Database) -> None:
    repository = RunRepository(database)
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    start_snapshot = _state(run.id)
    repository.create(run, start_snapshot)
    repository.start_stage(run.id, start_snapshot, "planner")

    event = repository.fail_stage(
        run.id,
        {**start_snapshot, "partial_result": "must not persist"},
        "planner",
        "provider_error",
        "failed",
    )

    stored = repository.get(run.id)
    assert stored is not None
    assert stored.status == "failed"
    assert stored.error_code == "provider_error"
    assert stored.error_message == "failed"
    assert repository.get_state(run.id) == start_snapshot
    assert event.event_type == "failed"


def test_list_incomplete_returns_only_queued_and_running_in_created_order(
    database: Database,
) -> None:
    repository = RunRepository(database)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    statuses = ["failed", "running", "completed", "queued"]
    runs = []
    for index, status in enumerate(statuses):
        run = ResearchRun.new(str(status), ["d1"], provider="fake", model="fake").model_copy(
            update={
                "id": f"run-{status}",
                "status": status,
                "created_at": base + timedelta(minutes=index),
                "updated_at": base + timedelta(minutes=index),
            }
        )
        repository.create(run, _state(run.id))
        runs.append(run)

    assert repository.list_incomplete() == [runs[1], runs[3]]


def test_two_threads_append_with_distinct_connections_and_allocate_inside_transaction(
    tmp_path: Path,
) -> None:
    class TrackingDatabase(Database):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.connections = []
            self.sequence_checks = []
            self.lock = threading.Lock()

        def connect(self):
            connection = super().connect()

            def trace(statement: str) -> None:
                if "MAX(sequence)" in statement:
                    with self.lock:
                        self.sequence_checks.append(connection.in_transaction)

            connection.set_trace_callback(trace)
            with self.lock:
                self.connections.append(connection)
            return connection

    database = TrackingDatabase(tmp_path / "threaded.db")
    database.initialize()
    repository = RunRepository(database)
    runs = [ResearchRun.new(f"q-{index}", ["d1"], "fake", "fake") for index in range(2)]
    for run in runs:
        repository.create(run, _state(run.id))
    database.connections.clear()
    database.sequence_checks.clear()

    with ThreadPoolExecutor(max_workers=2) as executor:
        events = list(
            executor.map(
                lambda run: repository.append_event(run.id, "planner", "retry", {"n": 1}),
                runs,
            )
        )

    assert [event.sequence for event in events] == [2, 2]
    assert len(database.connections) == 2
    assert database.connections[0] is not database.connections[1]
    assert database.sequence_checks == [True, True]
    for run in runs:
        assert [event.sequence for event in repository.list_events(run.id, 0)] == [1, 2]


def test_read_method_closes_its_fresh_connection(tmp_path: Path) -> None:
    class TrackingDatabase(Database):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.connections = []

        def connect(self):
            connection = super().connect()
            self.connections.append(connection)
            return connection

    database = TrackingDatabase(tmp_path / "reads.db")
    database.initialize()
    repository = RunRepository(database)
    run = ResearchRun.new("question", ["d1"], "fake", "fake")
    repository.create(run, _state(run.id))
    database.connections.clear()

    assert repository.get(run.id) == run

    assert len(database.connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        database.connections[0].execute("SELECT 1")
