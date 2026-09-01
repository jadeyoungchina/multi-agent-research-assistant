from types import SimpleNamespace
from typing import Any

from app.agents.planner import PlannerAgent
from app.agents.prompts import planner_messages
from app.domain.providers import ProviderMetadata
from app.domain.research import ResearchPlan
from app.providers.fake import FakeChatProvider
from app.providers.openai_compatible import OpenAICompatibleChatProvider
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


class JsonRequiredPlannerCompletions:
    """Fake JSON-mode API that rejects planner prompts lacking a JSON request."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        prompt = "\n".join(message["content"] for message in kwargs["messages"])
        if "json" not in prompt.lower():
            raise RuntimeError("JSON mode requires an explicit JSON instruction")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"objective":"compare","subquestions":["what agrees?"],'
                            '"search_queries":["agreement"],'
                            '"completion_criteria":["cite every finding"]}'
                        )
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
            model="qwen3.7-flash",
        )


def test_planner_succeeds_through_json_mode_provider_without_prompt_changes() -> None:
    completions = JsonRequiredPlannerCompletions()
    provider = OpenAICompatibleChatProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        provider_name="dashscope",
        model="qwen3.7-flash",
        max_retries=0,
    )
    question = "How do the studies compare?"

    update = PlannerAgent(provider)(initial_state("run", question, ["doc"]))

    assert update["plan"] == {
        "objective": "compare",
        "subquestions": ["what agrees?"],
        "search_queries": ["agreement"],
        "completion_criteria": ["cite every finding"],
    }
    assert completions.calls[0]["messages"][:2] == [
        message.model_dump() for message in planner_messages(question)
    ]
