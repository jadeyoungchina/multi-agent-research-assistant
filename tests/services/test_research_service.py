from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.domain.documents import DocumentRecord, EvidenceChunk
from app.domain.errors import WorkflowError
from app.domain.research import (
    Critique,
    DraftReport,
    Finding,
    ReportFinding,
    ResearchReport,
    ResearchSynthesis,
)
from app.storage.database import Database
from app.storage.documents import DocumentRepository
from app.storage.runs import RunRepository
from app.services.research import ResearchService
from app.workflow.checkpoints import checkpointed_node
from app.workflow.citations import CitationValidatorNode
from app.workflow.graph import WorkflowDependencies, build_research_graph
from app.workflow.routing import route_after_critic


def _document(document_id: str = "doc1", status: str = "ready") -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        filename=f"{document_id}.md",
        media_type="text/markdown",
        sha256=f"sha-{document_id}",
        storage_path=f"uploads/{document_id}.md",
        status=status,
        page_count=1,
        created_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
    )


def _report(evidence_sufficient: bool = True) -> ResearchReport:
    limitation = [] if evidence_sufficient else ["证据不足：Evidence remains incomplete"]
    return ResearchReport(
        title="Report",
        summary="Summary",
        findings=[],
        limitations=limitation,
        citations=[],
        markdown="# Report",
        evidence_sufficient=evidence_sufficient,
    )


class CompletingGraph:
    def __init__(
        self,
        repository: RunRepository,
        report: ResearchReport | None = None,
    ) -> None:
        self.repository = repository
        self.report = report or _report()
        self.calls = 0
        self.configs: list[dict] = []

    def invoke(self, state, config):
        self.calls += 1
        self.configs.append(config)

        def finish(current):
            del current
            return {
                "report": self.report.model_dump(mode="json"),
                "next_stage": "completed",
            }

        return checkpointed_node(
            "citation_validator", finish, self.repository
        )(state)


class FailingGraph:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, state, config):
        del state, config
        self.calls += 1
        raise RuntimeError("sensitive provider response")


class NonPersistingGraph:
    def invoke(self, state, config):
        del config
        return {**state, "next_stage": "completed"}


@pytest.fixture
def repositories(tmp_path: Path) -> tuple[DocumentRepository, RunRepository]:
    database = Database(tmp_path / "app.db")
    database.initialize()
    documents = DocumentRepository(database)
    documents.add_document(_document())
    return documents, RunRepository(database)


def _service(
    repositories: tuple[DocumentRepository, RunRepository], graph
) -> ResearchService:
    documents, runs = repositories
    return ResearchService(documents, runs, graph, "fake", "fake-chat")


def test_create_run_rejects_blank_question(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    service = _service(repositories, NonPersistingGraph())

    with pytest.raises(WorkflowError) as raised:
        service.create_run("   ", ["doc1"])

    assert raised.value.code == "empty_question"


def test_create_run_rejects_empty_document_selection(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    service = _service(repositories, NonPersistingGraph())

    with pytest.raises(WorkflowError) as raised:
        service.create_run("question", [])

    assert raised.value.code == "document_not_found"


def test_create_run_requires_existing_documents(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    service = _service(repositories, NonPersistingGraph())

    with pytest.raises(WorkflowError) as raised:
        service.create_run("question", ["missing"])

    assert raised.value.code == "document_not_found"


def test_create_run_requires_ready_documents(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()
    documents = DocumentRepository(database)
    documents.add_document(_document(status="failed"))
    service = ResearchService(
        documents, RunRepository(database), NonPersistingGraph(), "fake", "fake-chat"
    )

    with pytest.raises(WorkflowError) as raised:
        service.create_run("question", ["doc1"])

    assert raised.value.code == "document_not_ready"


def test_create_run_normalizes_question_and_preserves_document_order(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    documents, _runs = repositories
    documents.add_document(_document("doc2"))
    service = _service(repositories, NonPersistingGraph())

    run = service.create_run("  question  ", ["doc2", "doc1"])

    assert run.question == "question"
    assert run.document_ids == ["doc2", "doc1"]
    assert service.get_run(run.id) == run


def test_execute_is_idempotent_after_completion_and_uses_recursion_limit(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    graph = CompletingGraph(repositories[1])
    service = _service(repositories, graph)
    run = service.create_run("question", ["doc1"])

    first = service.execute(run.id)
    second = service.execute(run.id)

    assert second == first
    assert graph.calls == 1
    assert graph.configs == [{"recursion_limit": 24}]


def test_execute_marks_uncheckpointed_failure_without_leaking_message(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    graph = FailingGraph()
    service = _service(repositories, graph)
    run = service.create_run("question", ["doc1"])

    with pytest.raises(RuntimeError, match="sensitive provider response"):
        service.execute(run.id)

    stored = service.get_run(run.id)
    assert stored.status == "failed"
    assert stored.error_code == "workflow_failed"
    assert stored.error_message == "RuntimeError"
    assert [event.event_type for event in service.list_events(run.id)].count(
        "failed"
    ) == 1


def test_execute_refuses_failed_run_without_invoking_graph(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    graph = FailingGraph()
    service = _service(repositories, graph)
    run = service.create_run("question", ["doc1"])
    repositories[1].fail_stage(
        run.id,
        repositories[1].get_state(run.id),
        "planner",
        "workflow_failed",
        "RuntimeError",
    )

    with pytest.raises(WorkflowError) as raised:
        service.execute(run.id)

    assert raised.value.code == "run_failed"
    assert graph.calls == 0


def test_execute_requires_persisted_report(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    service = _service(repositories, NonPersistingGraph())
    run = service.create_run("question", ["doc1"])

    with pytest.raises(WorkflowError) as raised:
        service.execute(run.id)

    assert raised.value.code == "report_not_persisted"


@pytest.mark.parametrize(
    "method", ["execute", "get_run", "get_report", "list_events"]
)
def test_missing_run_raises_workflow_error(
    repositories: tuple[DocumentRepository, RunRepository], method: str
) -> None:
    service = _service(repositories, NonPersistingGraph())

    with pytest.raises(WorkflowError) as raised:
        getattr(service, method)("missing")

    assert raised.value.code == "run_not_found"


def test_report_events_and_incomplete_run_queries(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    graph = CompletingGraph(repositories[1])
    service = _service(repositories, graph)
    completed = service.create_run("completed", ["doc1"])
    queued = service.create_run("queued", ["doc1"])
    assert service.get_report(queued.id) is None

    report = service.execute(completed.id)

    assert service.get_report(completed.id) == report
    assert service.list_incomplete_ids() == [queued.id]
    assert [event.sequence for event in service.list_events(completed.id, 1)] == [
        2,
        3,
        4,
    ]


def test_exhausted_critic_completes_with_persisted_insufficiency(
    repositories: tuple[DocumentRepository, RunRepository],
) -> None:
    documents, runs = repositories
    evidence = EvidenceChunk(
        id="ev-1",
        document_id="doc1",
        filename="doc1.md",
        page_number=None,
        chunk_index=0,
        content="Supported fact.",
        content_sha256="evidence-sha",
    )

    def planner(state):
        del state
        return {
            "plan": {
                "objective": "Answer",
                "subquestions": ["What is supported?"],
                "search_queries": ["supported fact"],
                "completion_criteria": ["Supported fact"],
            },
            "next_stage": "retriever",
        }

    def retriever(state):
        del state
        return {
            "evidence": [evidence.model_dump(mode="json")],
            "next_stage": "researcher",
        }

    def researcher(state):
        del state
        synthesis = ResearchSynthesis(
            findings=[
                Finding(
                    claim="Supported fact",
                    supporting_evidence_ids=["ev-1"],
                    confidence="medium",
                )
            ]
        )
        return {
            "synthesis": synthesis.model_dump(mode="json"),
            "next_stage": "critic",
        }

    def critic(state):
        next_stage, iteration = route_after_critic(
            int(state.get("iteration", 0)), False, 2
        )
        critique = Critique(
            sufficient=False,
            reason="Evidence remains incomplete",
            follow_up_queries=["more evidence"],
        )
        return {
            "critique": critique.model_dump(mode="json"),
            "iteration": iteration,
            "next_stage": next_stage,
        }

    def writer(state):
        del state
        draft = DraftReport(
            title="Report",
            summary="Summary",
            findings=[
                ReportFinding(
                    heading="Finding",
                    narrative="Supported fact",
                    evidence_ids=["ev-1"],
                )
            ],
            markdown="# Report\n\nSupported fact [[cite:ev-1]]",
        )
        return {
            "draft": draft.model_dump(mode="json"),
            "next_stage": "citation_validator",
        }

    dependencies = WorkflowDependencies(
        planner=planner,
        retriever=retriever,
        researcher=researcher,
        critic=critic,
        writer=writer,
        citation_validator=CitationValidatorNode(),
        max_iterations=2,
        node_wrapper=lambda stage, handler: checkpointed_node(stage, handler, runs),
    )
    service = ResearchService(
        documents,
        runs,
        build_research_graph(dependencies),
        "fake",
        "fake-chat",
    )
    run = service.create_run("question", ["doc1"])

    report = service.execute(run.id)

    stored = service.get_run(run.id)
    assert stored.status == "completed"
    assert stored.evidence_sufficient is False
    assert report.evidence_sufficient is False
    assert report.limitations == ["证据不足：Evidence remains incomplete"]
    assert service.get_report(run.id) == report
    assert [event.event_type for event in service.list_events(run.id)].count(
        "finished"
    ) == 1
