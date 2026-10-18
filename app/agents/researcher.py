# Concept-only from multi_agent_collaboration_system.ipynb cells 6 and 11-21.
# Source commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46.
# Changes: typed local evidence, strict ID validation, one optional accounted repair.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt.

from typing import Any

from app.agents.common import EvidenceLabels, combine_provider_metadata, generate_grounded
from app.agents.prompts import researcher_messages
from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError, ProviderError
from app.domain.providers import ChatMessage, ChatProvider, ProviderMetadata
from app.domain.research import ResearchPlan, ResearchSynthesis
from app.workflow.state import WorkflowState


class ResearcherAgent:
    def __init__(self, provider: ChatProvider, *, max_citation_repairs: int = 0) -> None:
        if max_citation_repairs not in (0, 1):
            raise ValueError("researcher citation repairs must be zero or one")
        self.provider = provider
        self._max_citation_repairs = max_citation_repairs

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        evidence = [EvidenceChunk.model_validate(item) for item in state.get("evidence", [])]
        labels = EvidenceLabels(evidence)
        original = researcher_messages(state["question"], ResearchPlan.model_validate(state["plan"]), labels.evidence)
        messages, completed = original, []
        allowed = {item.id for item in evidence}
        for repair in range(self._max_citation_repairs + 1):
            try:
                synthesis, metadata = generate_grounded(self.provider, messages, ResearchSynthesis, [item.id for item in labels.evidence])
            except ProviderError as error:
                if isinstance(error.metadata, ProviderMetadata) and completed:
                    error.metadata = error.metadata.model_copy(deep=True)
                    error.metadata.retries += int(repair > 0)
                    error.metadata = combine_provider_metadata([*completed, error.metadata])
                raise
            metadata = metadata.model_copy(deep=True)
            metadata.retries += int(repair > 0)
            completed.append(metadata)
            synthesis = labels.synthesis(synthesis)
            referenced = {identifier for finding in synthesis.findings
                          for identifier in [*finding.supporting_evidence_ids, *finding.conflicting_evidence_ids]}
            if referenced <= allowed:
                break
            if repair == self._max_citation_repairs:
                error = CitationError("unknown_evidence_id", "researcher referenced unknown evidence IDs")
                error.metadata = combine_provider_metadata(completed)
                raise error
            messages = [*original,
                        ChatMessage(role="assistant", content=labels.synthesis(synthesis, to_provider=True).model_dump_json()),
                        ChatMessage(role="user", content=(
                            "Some evidence ID fields are invalid. Return one corrected ResearchSynthesis. "
                            "supporting_evidence_ids and conflicting_evidence_ids contain ONLY exact IDs "
                            "from the supplied evidence, never quoted passages or Cite: prefixes. "
                            "Check the passage supporting each claim; remove unsupported findings rather "
                            "than selecting an arbitrary valid ID. Keep unresolved details explicit. "
                            f"Allowed IDs: {[item.id for item in labels.evidence]}."
                        ))]
        return {"synthesis": synthesis.model_dump(mode="json"), "next_stage": "critic",
                "provider_metrics": [*state.get("provider_metrics", []),
                                     *(item.model_dump(mode="json") for item in completed)]}
