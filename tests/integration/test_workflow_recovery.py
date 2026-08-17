from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.domain.documents import DocumentRecord, EvidenceChunk
from app.domain.research import (
    Critique,
    DraftReport,
    Finding,
    ReportFinding,
    ResearchSynthesis,
)
from app.services.research import ResearchService
from app.storage.database import Database
from app.storage.documents import DocumentRepository
from app.storage.runs import RunRepository
from app.workflow.checkpoints import checkpointed_node
from app.workflow.citations import CitationValidatorNode
from app.workflow.graph import WorkflowDependencies, build_research_graph
from app.workflow.state import merge_state


@dataclass
class AppHarness:
    research: ResearchService
    run_repository: RunRepository
    ready_document: DocumentRecord
    calls: Counter


def _ready_document() -> DocumentRecord:
    return DocumentRecord(
        id="doc1",
        filename="source.md",
        media_type="text/markdown",
        sha256="source-sha",
        storage_path="uploads/source.md",
        status="ready",
        page_count=1,
        created_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
    )


def _build_app(root: Path) -> AppHarness:
    database = Database(root / "app.db")
    database.initialize()
    documents = DocumentRepository(database)
    ready_document = documents.get_document("doc1")
    if ready_document is None:
        ready_document = _ready_document()
        documents.add_document(ready_document)
    runs = RunRepository(database)
    calls = Counter()
    evidence = EvidenceChunk(
        id="ev-1",
        document_id=ready_document.id,
        filename=ready_document.filename,
        page_number=None,
        chunk_index=0,
        content="Supported fact.",
        content_sha256="evidence-sha",
    )

    def counted(stage, handler):
        checkpoint = checkpointed_node(stage, handler, runs)

        def invoke(state):
            calls[stage] += 1
            return checkpoint(state)

        return invoke

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
                    confidence="high",
                )
            ]
        )
        return {
            "synthesis": synthesis.model_dump(mode="json"),
            "next_stage": "critic",
        }

    def critic(state):
        del state
        critique = Critique(sufficient=True, reason="Enough evidence")
        return {
            "critique": critique.model_dump(mode="json"),
            "next_stage": "writer",
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

    graph = build_research_graph(
        WorkflowDependencies(
            planner=planner,
            retriever=retriever,
            researcher=researcher,
            critic=critic,
            writer=writer,
            citation_validator=CitationValidatorNode(),
            node_wrapper=counted,
        )
    )
    return AppHarness(
        research=ResearchService(documents, runs, graph, "fake", "fake-chat"),
        run_repository=runs,
        ready_document=ready_document,
        calls=calls,
    )


def _seed_after_retriever(app: AppHarness, run_id: str) -> None:
    state = app.run_repository.get_state(run_id)
    planner_update = {
        "plan": {
            "objective": "Answer",
            "subquestions": ["What is supported?"],
            "search_queries": ["supported fact"],
            "completion_criteria": ["Supported fact"],
        },
        "next_stage": "retriever",
    }
    app.run_repository.start_stage(run_id, state, "planner")
    state = merge_state(state, planner_update)
    state["last_completed_stage"] = "planner"
    app.run_repository.complete_stage(
        run_id, state, "retriever", "planner", 0, [], {}
    )

    evidence = EvidenceChunk(
        id="ev-1",
        document_id=app.ready_document.id,
        filename=app.ready_document.filename,
        page_number=None,
        chunk_index=0,
        content="Supported fact.",
        content_sha256="evidence-sha",
    )
    app.run_repository.start_stage(run_id, state, "retriever")
    state = merge_state(
        state,
        {
            "evidence": [evidence.model_dump(mode="json")],
            "next_stage": "researcher",
        },
    )
    state["last_completed_stage"] = "retriever"
    app.run_repository.complete_stage(
        run_id, state, "researcher", "retriever", 0, [], {}
    )


def test_restart_resumes_running_checkpoint_without_repeating_completed_nodes(
    tmp_path: Path,
) -> None:
    first = _build_app(tmp_path)
    run = first.research.create_run("question", [first.ready_document.id])
    _seed_after_retriever(first, run.id)

    second = _build_app(tmp_path)
    second.research.execute(run.id)

    assert second.calls["planner"] == 0
    assert second.calls["retriever"] == 0
    assert second.calls["researcher"] == 1
    assert second.calls["critic"] == 1
    assert second.calls["writer"] == 1
    assert second.calls["citation_validator"] == 1
    assert second.research.get_report(run.id) is not None
