from typing import Any

from app.agents.prompts import writer_messages
from app.domain.documents import EvidenceChunk
from app.domain.providers import ChatProvider
from app.domain.research import Critique, DraftReport, ResearchSynthesis
from app.workflow.citations import validate_draft_citations
from app.workflow.state import WorkflowState


def _preserve_insufficiency(draft: DraftReport, critique: Critique) -> DraftReport:
    if critique.sufficient:
        return draft

    reason = critique.reason.strip() or "现有证据未达到充分性要求。"
    if any(reason in limitation for limitation in draft.limitations):
        return draft

    updated = draft.model_copy(deep=True)
    updated.limitations.append(f"证据不足：{reason}")
    return updated


class WriterAgent:
    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider

    def __call__(self, state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        evidence = [EvidenceChunk.model_validate(item) for item in state["evidence"]]
        synthesis = ResearchSynthesis.model_validate(state["synthesis"])
        critique = Critique.model_validate(state["critique"])
        draft, metadata = self._provider.generate_structured(
            writer_messages(state["question"], synthesis, critique, evidence),
            DraftReport,
        )
        draft = _preserve_insufficiency(draft, critique)
        validate_draft_citations(draft, evidence)

        provider_metrics = list(state.get("provider_metrics", []))
        provider_metrics.append(metadata.model_dump(mode="json"))
        return {
            "draft": draft.model_dump(mode="json"),
            "next_stage": "citation_validator",
            "provider_metrics": provider_metrics,
        }
