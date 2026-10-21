from typing import Any

from app.agents.common import EvidenceLabels, combine_provider_metadata, generate_grounded
from app.agents.prompts import writer_messages
from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError, ProviderError
from app.domain.providers import ChatMessage, ChatProvider, ProviderMetadata
from app.domain.research import Critique, DraftReport, ResearchSynthesis
from app.workflow.citations import validate_draft_citations, report_limitations
from app.workflow.state import WorkflowState


def _preserve_insufficiency(draft: DraftReport, critique: Critique, question: str) -> DraftReport:
    limitations = report_limitations(draft, critique, question=question)
    if limitations == draft.limitations:
        return draft
    updated = draft.model_copy(deep=True)
    updated.limitations = limitations
    return updated


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
        labels = EvidenceLabels(evidence)
        original = writer_messages(state["question"], labels.synthesis(synthesis, to_provider=True), critique, labels.evidence)
        messages = original
        completed = []
        for repair in range(self._max_citation_repairs + 1):
            try:
                draft, metadata = generate_grounded(self._provider, messages, DraftReport, [item.id for item in labels.evidence])
            except ProviderError as error:
                if isinstance(error.metadata, ProviderMetadata) and completed:
                    error.metadata = error.metadata.model_copy(deep=True)
                    error.metadata.retries += int(repair > 0)
                    error.metadata = combine_provider_metadata([*completed, error.metadata])
                raise
            metadata = metadata.model_copy(deep=True)
            metadata.retries += int(repair > 0)  # Includes explicit citation repair, separately bounded.
            completed.append(metadata)
            draft = labels.draft(draft)
            try:
                draft = _preserve_insufficiency(draft, critique, state["question"])
                validate_draft_citations(draft, evidence)
                break
            except CitationError as error:
                if repair == self._max_citation_repairs:
                    # Invalid drafts are not committed, but all paid calls count.
                    error.metadata = combine_provider_metadata(completed)
                    raise
                messages = [*original, ChatMessage(role="assistant", content=labels.draft(draft, to_provider=True).model_dump_json()),
                            ChatMessage(role="user", content=(
                                f"Citation validation failed: {error.code}. Revise the JSON report once. "
                                "The markdown string is the user-visible answer. Put exact [[cite:ID]] "
                                "tokens next to supported claims INSIDE markdown; citations only in "
                                "summary or findings.narrative do not satisfy this. Every finding "
                                "evidence_id must appear in markdown. Use only supplied evidence IDs. "
                                f"Allowed IDs: {[chunk.id for chunk in labels.evidence]}. "
                                "Do not invent evidence or weaken the answer to bypass validation."
                            ))]

        provider_metrics = list(state.get("provider_metrics", []))
        provider_metrics.extend(item.model_dump(mode="json") for item in completed)
        return {
            "draft": draft.model_dump(mode="json"),
            "next_stage": "citation_validator",
            "provider_metrics": provider_metrics,
        }
