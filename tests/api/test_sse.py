import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import ApplicationServices
from app.api.sse import encode_sse, stream_run_events
from app.domain.errors import WorkflowError
from app.domain.runs import ResearchRun, RunEvent
from app.main import create_app


def _run(status: str = "completed") -> ResearchRun:
    now = datetime(2026, 9, 10, tzinfo=UTC)
    return ResearchRun(
        id="run1",
        question="What does the evidence show?",
        document_ids=["doc1"],
        status=status,
        current_stage="completed" if status == "completed" else "planner",
        iteration=0,
        provider="fake",
        model="fake-chat",
        created_at=now,
        updated_at=now,
        completed_at=now if status in {"completed", "failed"} else None,
    )


def _event(sequence: int) -> RunEvent:
    return RunEvent(
        sequence=sequence,
        run_id="run1",
        stage="planner",
        event_type="created",
        payload={"question": "What does the evidence show?"},
        created_at=datetime(2026, 9, 10, tzinfo=UTC),
    )


class FakeResearchService:
    def __init__(self) -> None:
        self.run = _run()
        self.events = [_event(sequence) for sequence in range(1, 5)]

    def list_incomplete_ids(self) -> list[str]:
        return []

    def get_run(self, run_id: str) -> ResearchRun:
        if run_id != "run1":
            raise WorkflowError("run_not_found", "research run does not exist")
        return self.run

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]:
        self.get_run(run_id)
        return [event for event in self.events if event.sequence > after_sequence]


class FakeTaskManager:
    def start(self, run_id: str) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def event() -> RunEvent:
    return _event(7)


@pytest.fixture
def client() -> Iterator[TestClient]:
    services = ApplicationServices(
        documents=SimpleNamespace(), research=FakeResearchService()
    )
    with TestClient(
        create_app(services=services, task_manager=FakeTaskManager()),
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


def test_encode_sse_has_id_event_and_compact_json(event: RunEvent) -> None:
    frame = encode_sse(event).decode("utf-8")

    assert frame.startswith("id: 7\nevent: workflow\n")
    assert 'data: {"sequence":7' in frame
    assert frame.endswith("\n\n")


def test_event_stream_resumes_after_larger_last_event_id(client: TestClient) -> None:
    with client.stream(
        "GET", "/api/research/run1/events?after=2", headers={"Last-Event-ID": "3"}
    ) as response:
        payload = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "id: 3\n" not in payload
    assert "id: 4\n" in payload


def test_event_stream_returns_not_found_before_opening_stream(client: TestClient) -> None:
    response = client.get("/api/research/missing/events")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "run_not_found"
    assert response.headers["x-request-id"]


def test_event_generator_emits_heartbeat_while_run_is_active() -> None:
    service = FakeResearchService()
    service.run = _run(status="running")

    class RequestThatDisconnectsAfterHeartbeat:
        def __init__(self) -> None:
            self.calls = 0

        async def is_disconnected(self) -> bool:
            self.calls += 1
            return self.calls > 1

    async def read_heartbeat() -> bytes:
        generator = stream_run_events(
            RequestThatDisconnectsAfterHeartbeat(), "run1", service, 4, 0, 0
        )
        return await anext(generator)

    assert asyncio.run(read_heartbeat()) == b": keep-alive\n\n"


def test_event_generator_stops_without_querying_after_disconnect() -> None:
    service = FakeResearchService()
    calls: list[str] = []

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            calls.append("disconnect")
            return True

    async def read_events() -> list[bytes]:
        return [
            event
            async for event in stream_run_events(
                DisconnectedRequest(), "run1", service, 0, 0, 0
            )
        ]

    assert asyncio.run(read_events()) == []
    assert calls == ["disconnect"]


def test_event_generator_rechecks_events_committed_during_terminal_transition() -> None:
    class CompletionRaceService(FakeResearchService):
        def __init__(self) -> None:
            super().__init__()
            self.events = []
            self.run = _run(status="running")
            self.list_calls = 0

        def list_events(
            self, run_id: str, after_sequence: int = 0
        ) -> list[RunEvent]:
            self.list_calls += 1
            return [event for event in self.events if event.sequence > after_sequence]

        def get_run(self, run_id: str) -> ResearchRun:
            if self.list_calls == 1:
                self.events.append(_event(1))
                self.run = _run(status="completed")
            return super().get_run(run_id)

    async def read_events() -> list[bytes]:
        return [
            event
            async for event in stream_run_events(
                RequestThatStaysConnected(), "run1", CompletionRaceService(), 0, 0, 60
            )
        ]

    assert asyncio.run(read_events()) == [encode_sse(_event(1))]


class RequestThatStaysConnected:
    async def is_disconnected(self) -> bool:
        return False
