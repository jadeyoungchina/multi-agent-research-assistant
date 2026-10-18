from typing import Any

from app.agents.prompts import writer_messages
from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError, ProviderError
from app.domain.providers import ChatMessage, ChatProvider, ProviderMetadata, TokenUsage
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


def _combined_metadata(items: list[ProviderMetadata]) -> ProviderMetadata:
    combined = items[-1].model_copy(deep=True)
    combined.usage = TokenUsage(**{
        field: sum(getattr(item.usage, field) for item in items)
        for field in TokenUsage.model_fields
    })
    combined.latency_ms = sum(item.latency_ms for item in items)
    combined.retries = sum(item.retries for item in items)
    combined.attempt_count = sum(item.attempt_count for item in items)
    combined.attempt_ids = [identifier for item in items for identifier in item.attempt_ids]
    combined.usage_complete = all(item.usage_complete for item in items)
    return combined


class WriterAgent:
    def __init__(self, provider: ChatProvider, *, max_citation_repairs: int = 0) -> None:
        if max_citation_repairs not in (0, 1):
            raise ValueError("writer citation repairs must be zero or one")
        self._provider = provider
        self._max_citation_repairs = max_citation_repairs

    def __call__(self, state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        evidence = [EvidenceChunk.model_validate(item) for item in state["evidence"]]
        synthesis = ResearchSynthesis.model_validate(state["synthesis"])
        critique = Critique.model_validate(state["critique"])
        original = writer_messages(state["question"], synthesis, critique, evidence)
        messages = original
        completed = []
        for repair in range(self._max_citation_repairs + 1):
            try:
                draft, metadata = self._provider.generate_structured(messages, DraftReport)
            except ProviderError as error:
                if isinstance(error.metadata, ProviderMetadata) and completed:
                    error.metadata = error.metadata.model_copy(deep=True)
                    error.metadata.retries += int(repair > 0)
                    error.metadata = _combined_metadata([*completed, error.metadata])
                raise
            metadata = metadata.model_copy(deep=True)
            metadata.retries += int(repair > 0)  # Includes explicit citation repair, separately bounded.
            completed.append(metadata)
            try:
                draft = _preserve_insufficiency(draft, critique)
                validate_draft_citations(draft, evidence)
                break
            except CitationError as error:
                if repair == self._max_citation_repairs:
                    # Invalid drafts are not committed, but all paid calls count.
                    error.metadata = _combined_metadata(completed)
                    raise
                messages = [*original, ChatMessage(role="assistant", content=draft.model_dump_json()),
                            ChatMessage(role="user", content=(
                                f"Citation validation failed: {error.code}. Revise the JSON report once. "
                                "The markdown string is the user-visible answer. Put exact [[cite:ID]] "
                                "tokens next to supported claims INSIDE markdown; citations only in "
                                "summary or findings.narrative do not satisfy this. Every finding "
                                "evidence_id must appear in markdown. Use only supplied evidence IDs. "
                                "Do not invent evidence or weaken the answer to bypass validation."
                            ))]

        provider_metrics = list(state.get("provider_metrics", []))
        provider_metrics.extend(item.model_dump(mode="json") for item in completed)
        return {
            "draft": draft.model_dump(mode="json"),
            "next_stage": "citation_validator",
            "provider_metrics": provider_metrics,
        }
