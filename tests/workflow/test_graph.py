from collections.abc import Callable

import pytest
from langgraph.errors import GraphRecursionError

from app.domain.documents import EvidenceChunk
from app.domain.research import (
    Critique,
    DraftReport,
    Finding,
    ReportFinding,
    ResearchSynthesis,
)
from app.workflow.citations import CitationValidatorNode
from app.workflow.graph import WorkflowDependencies, build_research_graph
from app.workflow.routing import route_after_critic
from app.workflow.state import WorkflowState, initial_state


Node = Callable[[WorkflowState], dict[str, object]]


class FakeWorkflow:
    def __init__(self, critic_results: list[bool] | None = None) -> None:
        self.critic_results = list(critic_results or [True])
        self.visited: list[str] = []
        self.max_iterations = 2

    def wrap(self, stage: str, handler: Node) -> Node:
        def record_visit(state: WorkflowState) -> dict[str, object]:
            self.visited.append(stage)
            return handler(state)

        return record_visit

    def planner(self, state: WorkflowState) -> dict[str, object]:
        del state
        return {"plan": {"objective": "Answer"}, "next_stage": "retriever"}

    def retriever(self, state: WorkflowState) -> dict[str, object]:
        del state
        evidence = EvidenceChunk(
            id="ev-1",
            document_id="doc",
            filename="source.md",
            page_number=None,
            chunk_index=0,
            content="Supported fact.",
            content_sha256="abc123",
        )
        return {
            "evidence": [evidence.model_dump(mode="json")],
            "next_stage": "researcher",
        }

    def researcher(self, state: WorkflowState) -> dict[str, object]:
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

    def critic(self, state: WorkflowState) -> dict[str, object]:
        if not self.critic_results:
            raise AssertionError("no fake critic result queued")
        sufficient = self.critic_results.pop(0)
        next_stage, iteration = route_after_critic(
            state.get("iteration", 0), sufficient, self.max_iterations
        )
        critique = Critique(
            sufficient=sufficient,
            reason=("Enough evidence" if sufficient else "Evidence remains incomplete"),
            follow_up_queries=[] if sufficient else ["find more evidence"],
        )
        return {
            "critique": critique.model_dump(mode="json"),
            "iteration": iteration,
            "next_stage": next_stage,
        }

    def writer(self, state: WorkflowState) -> dict[str, object]:
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

    def citation_validator(self, state: WorkflowState) -> dict[str, object]:
        return CitationValidatorNode()(state)

    def dependencies(self) -> WorkflowDependencies:
        return WorkflowDependencies(
            planner=self.planner,
            retriever=self.retriever,
            researcher=self.researcher,
            critic=self.critic,
            writer=self.writer,
            citation_validator=self.citation_validator,
            max_iterations=self.max_iterations,
            node_wrapper=self.wrap,
        )


@pytest.fixture
def workflow() -> FakeWorkflow:
    return FakeWorkflow()


def seeded_state(next_stage: str) -> WorkflowState:
    state = initial_state("run", "question", ["doc"])
    state.update(FakeWorkflow().retriever(state))
    state.update(FakeWorkflow().researcher(state))
    state["critique"] = Critique(
        sufficient=True, reason="Enough evidence"
    ).model_dump(mode="json")
    state.update(FakeWorkflow().writer(state))
    state["next_stage"] = next_stage
    return state


def test_graph_runs_happy_path_once(workflow: FakeWorkflow) -> None:
    result = build_research_graph(workflow.dependencies()).invoke(
        initial_state("run", "question", ["doc"]), {"recursion_limit": 24}
    )

    assert workflow.visited == [
        "planner",
        "retriever",
        "researcher",
        "critic",
        "writer",
        "citation_validator",
    ]
    assert result["next_stage"] == "completed"


def test_graph_runs_one_revision_before_writing() -> None:
    workflow = FakeWorkflow([False, True])

    result = build_research_graph(workflow.dependencies()).invoke(
        initial_state("run", "question", ["doc"]), {"recursion_limit": 24}
    )

    assert workflow.visited == [
        "planner",
        "retriever",
        "researcher",
        "critic",
        "retriever",
        "researcher",
        "critic",
        "writer",
        "citation_validator",
    ]
    assert result["iteration"] == 1
    assert Critique.model_validate(result["critique"]).sufficient is True


def test_graph_retrieves_three_times_when_critic_exhausts_two_revisions() -> None:
    workflow = FakeWorkflow([False, False, False])

    result = build_research_graph(workflow.dependencies()).invoke(
        initial_state("run", "question", ["doc"]), {"recursion_limit": 24}
    )

    critique = Critique.model_validate(result["critique"])
    assert workflow.visited.count("retriever") == 3
    assert workflow.visited == [
        "planner",
        "retriever",
        "researcher",
        "critic",
        "retriever",
        "researcher",
        "critic",
        "retriever",
        "researcher",
        "critic",
        "writer",
        "citation_validator",
    ]
    assert result["iteration"] == 2
    assert critique.sufficient is False
    assert result["report"]["evidence_sufficient"] is False
    assert result["report"]["limitations"] == [
        "证据不足：Evidence remains incomplete"
    ]


def test_graph_resume_at_researcher_skips_planner_and_retriever() -> None:
    workflow = FakeWorkflow([True])

    result = build_research_graph(workflow.dependencies()).invoke(
        seeded_state("researcher"), {"recursion_limit": 24}
    )

    assert workflow.visited == [
        "researcher",
        "critic",
        "writer",
        "citation_validator",
    ]
    assert result["next_stage"] == "completed"


def test_graph_routes_directly_from_sufficient_critic_to_writer() -> None:
    workflow = FakeWorkflow([True])
    state = seeded_state("critic")
    state["iteration"] = 2

    result = build_research_graph(workflow.dependencies()).invoke(
        state, {"recursion_limit": 24}
    )

    assert workflow.visited == ["critic", "writer", "citation_validator"]
    assert result["iteration"] == 2
    assert result["next_stage"] == "completed"


def test_graph_accepts_completed_state_without_running_a_node() -> None:
    workflow = FakeWorkflow()
    state = seeded_state("completed")

    result = build_research_graph(workflow.dependencies()).invoke(
        state, {"recursion_limit": 24}
    )

    assert workflow.visited == []
    assert result["next_stage"] == "completed"


def test_graph_raises_when_recursion_limit_is_too_low() -> None:
    workflow = FakeWorkflow([False, False, False])

    with pytest.raises(GraphRecursionError):
        build_research_graph(workflow.dependencies()).invoke(
            initial_state("run", "question", ["doc"]), {"recursion_limit": 4}
        )
