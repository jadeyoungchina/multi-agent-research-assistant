import logging
from pathlib import Path

import pytest

from app.domain.errors import WorkflowError
from app.domain.runs import ResearchRun
from app.storage.database import Database
from app.storage.runs import RunRepository
from app.workflow.checkpoints import checkpointed_node
from app.workflow.state import initial_state


@pytest.fixture
def repository(tmp_path: Path) -> RunRepository:
    database = Database(tmp_path / "app.db")
    database.initialize()
    value = RunRepository(database)
    run = ResearchRun.new("question", ["doc"], provider="fake", model="fake")
    run = run.model_copy(update={"id": "run"})
    value.create(run, initial_state(run.id, run.question, run.document_ids))
    return value


def test_completed_stage_saves_merged_state_and_event(
    repository: RunRepository,
) -> None:
    def handler(state):
        del state
        return {"plan": {"objective": "x"}, "next_stage": "retriever"}

    node = checkpointed_node("planner", handler, repository)

    update = node(initial_state("run", "question", ["doc"]))

    stored = repository.get_state("run")
    assert update["last_completed_stage"] == "planner"
    assert stored["plan"]["objective"] == "x"
    assert stored["last_completed_stage"] == "planner"
    assert [event.event_type for event in repository.list_events("run", 0)][-2:] == [
        "started",
        "completed",
    ]


def test_failed_stage_keeps_previous_state_and_stores_only_exception_type(
    repository: RunRepository,
) -> None:
    before = repository.get_state("run")

    def handler(state):
        state["partial_result"] = "must not persist"
        raise WorkflowError("provider_failed", "secret provider response")

    node = checkpointed_node("planner", handler, repository)

    with pytest.raises(WorkflowError, match="secret provider response"):
        node(before)

    stored = repository.get("run")
    assert stored is not None
    assert repository.get_state("run") == initial_state(
        "run", "question", ["doc"]
    )
    assert stored.status == "failed"
    assert stored.error_code == "workflow_stage_failed"
    assert stored.error_message == "WorkflowError"


def test_checkpoint_logs_structured_safe_metadata(
    repository: RunRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = initial_state("run", "question", ["doc"])
    state["provider_metrics"] = [
        {
            "provider": "fake",
            "model": "fake-chat",
            "latency_ms": 7,
            "retries": 1,
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
            "secret": "must-not-be-logged",
        }
    ]

    def handler(current):
        del current
        raise RuntimeError("sensitive failure detail")

    caplog.set_level(logging.INFO, logger="app.workflow.checkpoints")

    with pytest.raises(RuntimeError, match="sensitive failure detail"):
        checkpointed_node("planner", handler, repository)(state)

    records = [
        record
        for record in caplog.records
        if record.name == "app.workflow.checkpoints"
    ]
    assert [record.getMessage() for record in records] == [
        "workflow_stage_started",
        "workflow_stage_failed",
    ]
    assert all(record.run_id == "run" for record in records)
    assert all(record.stage == "planner" for record in records)
    assert all(isinstance(record.elapsed_ms, int) for record in records)
    assert records[-1].exception_type == "RuntimeError"
    assert records[-1].provider_metrics == [
        {
            "provider": "fake",
            "model": "fake-chat",
            "latency_ms": 7,
            "retries": 1,
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }
    ]
    assert "sensitive failure detail" not in caplog.text
    assert "must-not-be-logged" not in caplog.text


def test_completed_stage_logs_updated_safe_provider_metrics(
    repository: RunRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    metric = {
        "provider": "fake",
        "model": "fake-chat",
        "latency_ms": 9,
        "retries": 0,
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "total_tokens": 7,
        },
    }

    def handler(state):
        del state
        return {
            "next_stage": "retriever",
            "provider_metrics": [{**metric, "unsafe": "not logged"}],
        }

    caplog.set_level(logging.INFO, logger="app.workflow.checkpoints")

    checkpointed_node("planner", handler, repository)(
        initial_state("run", "question", ["doc"])
    )

    records = [
        record
        for record in caplog.records
        if record.name == "app.workflow.checkpoints"
    ]
    assert [record.getMessage() for record in records] == [
        "workflow_stage_started",
        "workflow_stage_completed",
    ]
    assert records[-1].provider_metrics == [metric]
    assert "not logged" not in caplog.text
