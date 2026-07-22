# Adapted from scientific_paper_agent_langgraph.ipynb cells 15, 19, and 21.
# Source commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46.
# Changes: five explicit roles, local evidence retrieval, bounded revision,
# resume dispatch, deterministic citation validation, dependency injection.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt.

from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from app.domain.research import Critique
from app.workflow.routing import route_after_critic
from app.workflow.state import WorkflowState


Node = Callable[[WorkflowState], dict[str, Any]]
NodeWrapper = Callable[[str, Node], Node]


@dataclass(frozen=True)
class WorkflowDependencies:
    planner: Node
    retriever: Node
    researcher: Node
    critic: Node
    writer: Node
    citation_validator: Node
    max_iterations: int = 2
    node_wrapper: NodeWrapper | None = None


def _critic_route(state: WorkflowState, max_iterations: int) -> str:
    critique = Critique.model_validate(state["critique"])
    iteration = int(state.get("iteration", 0))
    chosen_stage = state["next_stage"]
    prior_iteration = (
        iteration - 1
        if not critique.sufficient and chosen_stage == "retriever"
        else iteration
    )
    next_stage, routed_iteration = route_after_critic(
        prior_iteration, critique.sufficient, max_iterations
    )
    if next_stage != chosen_stage or routed_iteration != iteration:
        raise ValueError("critic update does not match bounded revision routing")
    return next_stage


def build_research_graph(deps: WorkflowDependencies):
    graph = StateGraph(WorkflowState)
    wrap = deps.node_wrapper or (lambda _stage, handler: handler)

    graph.add_node("planner", wrap("planner", deps.planner))
    graph.add_node("retriever", wrap("retriever", deps.retriever))
    graph.add_node("researcher", wrap("researcher", deps.researcher))
    graph.add_node("critic", wrap("critic", deps.critic))
    graph.add_node("writer", wrap("writer", deps.writer))
    graph.add_node(
        "citation_validator",
        wrap("citation_validator", deps.citation_validator),
    )

    graph.set_conditional_entry_point(
        lambda state: state["next_stage"],
        {
            "planner": "planner",
            "retriever": "retriever",
            "researcher": "researcher",
            "critic": "critic",
            "writer": "writer",
            "citation_validator": "citation_validator",
            "completed": END,
        },
    )
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        lambda state: _critic_route(state, deps.max_iterations),
        {"retriever": "retriever", "writer": "writer"},
    )
    graph.add_edge("writer", "citation_validator")
    graph.add_edge("citation_validator", END)
    return graph.compile()
