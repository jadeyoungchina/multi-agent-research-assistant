from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import ApplicationServices
from app.domain.errors import WorkflowError
from app.domain.research import Citation, ReportFinding, ResearchReport
from app.domain.runs import ResearchRun
from app.main import create_app


def _run(
    run_id: str = "run1",
    status: str = "queued",
    evidence_sufficient: bool | None = None,
) -> ResearchRun:
    now = datetime(2026, 9, 10, tzinfo=UTC)
    return ResearchRun(
        id=run_id,
        question="What does the evidence show?",
        document_ids=["doc1"],
        status=status,
        current_stage="completed" if status == "completed" else "planner",
        iteration=2 if status == "completed" else 0,
        provider="fake",
        model="fake-chat",
        evidence_sufficient=evidence_sufficient,
        error_code="workflow_failed" if status == "failed" else None,
        error_message="RuntimeError" if status == "failed" else None,
        created_at=now,
        updated_at=now,
        completed_at=now if status in {"completed", "failed"} else None,
    )


def _report(evidence_sufficient: bool = True) -> ResearchReport:
    return ResearchReport(
        title="Evidence review",
        summary="Supported summary",
        findings=[
            ReportFinding(
                heading="Finding",
                narrative="Supported claim",
                evidence_ids=["ev_good"],
            )
        ],
        limitations=(
            [] if evidence_sufficient else ["证据不足：More evidence is needed"]
        ),
        citations=[
            Citation(
                evidence_id="ev_good",
                filename="paper.pdf",
                page_number=1,
                chunk_index=0,
                excerpt="Supporting evidence",
            )
        ],
        markdown="# Evidence review",
        evidence_sufficient=evidence_sufficient,
    )


class FakeResearchService:
    def __init__(self) -> None:
        self.created: list[tuple[str, list[str]]] = []
        self.runs: dict[str, ResearchRun] = {"run1": _run()}
        self.reports: dict[str, ResearchReport] = {}
        self.incomplete_ids: list[str] = []

    def create_run(self, question: str, document_ids: list[str]) -> ResearchRun:
        self.created.append((question, document_ids))
        return self.runs["run1"]

    def get_run(self, run_id: str) -> ResearchRun:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise WorkflowError(
                "run_not_found", "research run does not exist"
            ) from exc

    def get_report(self, run_id: str) -> ResearchReport | None:
        self.get_run(run_id)
        return self.reports.get(run_id)

    def list_incomplete_ids(self) -> list[str]:
        return list(self.incomplete_ids)


class RecordingTaskManager:
    def __init__(self, events: list[str] | None = None) -> None:
        self.started: list[str] = []
        self.close_count = 0
        self.events = events

    def start(self, run_id: str) -> None:
        self.started.append(run_id)

    def close(self) -> None:
        self.close_count += 1
        if self.events is not None:
            self.events.append("tasks")


@pytest.fixture
def research_service() -> FakeResearchService:
    return FakeResearchService()


@pytest.fixture
def task_manager() -> RecordingTaskManager:
    return RecordingTaskManager()


@pytest.fixture
def research_client(
    research_service: FakeResearchService,
    task_manager: RecordingTaskManager,
) -> Iterator[TestClient]:
    services = ApplicationServices(
        documents=SimpleNamespace(),
        research=research_service,
    )
    app = create_app(services=services, task_manager=task_manager)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_create_research_returns_202_and_location(
    research_client: TestClient,
    task_manager: RecordingTaskManager,
) -> None:
    response = research_client.post(
        "/api/research",
        json={
            "question": "What does the evidence show?",
            "document_ids": ["doc1"],
        },
    )

    assert response.status_code == 202
    assert response.headers["Location"] == "/api/research/run1"
    assert response.json() == {"run_id": "run1", "status": "queued"}
    assert task_manager.started == ["run1"]


def test_create_research_deduplicates_document_ids_in_first_seen_order(
    research_client: TestClient,
    research_service: FakeResearchService,
) -> None:
    response = research_client.post(
        "/api/research",
        json={
            "question": "What does the evidence show?",
            "document_ids": ["doc2", "doc1", "doc2", "doc3", "doc1"],
        },
    )

    assert response.status_code == 202
    assert research_service.created == [
        ("What does the evidence show?", ["doc2", "doc1", "doc3"])
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   ", "document_ids": ["doc1"]},
        {"question": "ok", "document_ids": ["doc1"]},
        {"question": "What does the evidence show?", "document_ids": []},
    ],
)
def test_create_research_rejects_invalid_requests(
    research_client: TestClient,
    task_manager: RecordingTaskManager,
    payload: dict,
) -> None:
    response = research_client.post("/api/research", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert task_manager.started == []


def test_get_completed_run_includes_report(
    research_client: TestClient,
    research_service: FakeResearchService,
) -> None:
    research_service.runs["run1"] = _run(
        status="completed", evidence_sufficient=True
    )
    research_service.reports["run1"] = _report()

    response = research_client.get("/api/research/run1")

    assert response.status_code == 200
    assert response.json()["run"]["status"] == "completed"
    assert response.json()["run"]["evidence_sufficient"] is True
    assert response.json()["report"]["citations"][0]["evidence_id"] == "ev_good"


def test_get_failed_run_includes_failure_status_without_report(
    research_client: TestClient,
    research_service: FakeResearchService,
) -> None:
    research_service.runs["run1"] = _run(status="failed")

    response = research_client.get("/api/research/run1")

    assert response.status_code == 200
    assert response.json()["run"]["status"] == "failed"
    assert response.json()["run"]["error_code"] == "workflow_failed"
    assert response.json()["run"]["error_message"] == "RuntimeError"
    assert response.json()["report"] is None


def test_get_insufficient_evidence_run_remains_completed_with_limitations(
    research_client: TestClient,
    research_service: FakeResearchService,
) -> None:
    research_service.runs["run1"] = _run(
        status="completed", evidence_sufficient=False
    )
    research_service.reports["run1"] = _report(evidence_sufficient=False)

    response = research_client.get("/api/research/run1")

    assert response.status_code == 200
    assert response.json()["run"]["status"] == "completed"
    assert response.json()["run"]["evidence_sufficient"] is False
    assert response.json()["report"]["evidence_sufficient"] is False
    assert response.json()["report"]["limitations"] == [
        "证据不足：More evidence is needed"
    ]


def test_get_unknown_run_returns_not_found(research_client: TestClient) -> None:
    response = research_client.get("/api/research/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "run_not_found"


def test_lifespan_recovers_each_incomplete_run_once_with_injected_manager(
    research_service: FakeResearchService,
    task_manager: RecordingTaskManager,
) -> None:
    research_service.incomplete_ids = ["run1", "run2", "run1"]
    services = ApplicationServices(
        documents=SimpleNamespace(),
        research=research_service,
    )
    app = create_app(services=services, task_manager=task_manager)

    with TestClient(app):
        assert app.state.run_tasks is task_manager
        assert task_manager.started == ["run1", "run2"]

    assert task_manager.close_count == 1


def test_lifespan_closes_task_manager_before_provider_clients(
    application_container,
) -> None:
    events: list[str] = []
    task_manager = RecordingTaskManager(events)
    application_container.services = ApplicationServices(
        documents=SimpleNamespace(),
        research=FakeResearchService(),
    )
    application_container.chat_provider.close = lambda: events.append("provider")
    app = create_app(container=application_container, task_manager=task_manager)

    with TestClient(app):
        pass

    assert events == ["tasks", "provider"]
