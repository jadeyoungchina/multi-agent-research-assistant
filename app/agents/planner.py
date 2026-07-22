# Adapted from scientific_paper_agent_langgraph.ipynb cells 13 and 19.
# Source commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46.
# Changes: provider injection, ResearchPlan schema, no global model, no direct answer shortcut.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt.

from typing import Any

from app.agents.prompts import planner_messages
from app.domain.providers import ChatProvider
from app.domain.research import ResearchPlan
from app.workflow.state import WorkflowState


class PlannerAgent:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        plan, metadata = self.provider.generate_structured(
            planner_messages(state["question"]), ResearchPlan
        )
        return {
            "plan": plan.model_dump(mode="json"),
            "next_stage": "retriever",
            "provider_metrics": [
                *state.get("provider_metrics", []),
                metadata.model_dump(mode="json"),
            ],
        }
