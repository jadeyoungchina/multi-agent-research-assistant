from app.agents.planner import PlannerAgent
from app.domain.providers import ProviderMetadata
from app.domain.research import ResearchPlan
from app.providers.fake import FakeChatProvider
from app.workflow.state import initial_state


def test_planner_stores_validated_plan_and_appends_one_metadata_item() -> None:
    """Removing the structured plan or metric append must break this test."""
    plan = ResearchPlan(
        objective="compare",
        subquestions=["what agrees?", "what conflicts?"],
        search_queries=["agreement", "conflict"],
        completion_criteria=["cite every finding"],
    )
    provider = FakeChatProvider(structured_responses=[plan])
    state = initial_state("run", "question", ["doc"])
    state["provider_metrics"] = [
        ProviderMetadata(provider="previous", model="old", latency_ms=1).model_dump(
            mode="json"
        )
    ]

    update = PlannerAgent(provider)(state)

    assert update["plan"] == plan.model_dump(mode="json")
    assert update["next_stage"] == "retriever"
    assert len(update["provider_metrics"]) == 2
    assert update["provider_metrics"][-1]["provider"] == "fake"
