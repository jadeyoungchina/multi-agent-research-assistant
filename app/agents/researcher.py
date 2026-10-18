# Concept-only from multi_agent_collaboration_system.ipynb cells 6 and 11-21.
# Source commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46.
# Changes: inject provider, synthesize against typed local evidence, and reject ungrounded citations deterministically.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt.

from typing import Any
import re

from app.agents.prompts import researcher_messages
from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError
from app.domain.providers import ChatProvider
from app.domain.research import ResearchPlan, ResearchSynthesis
from app.workflow.state import WorkflowState


def _canonical_evidence_id(value: str, allowed: set[str]) -> str:
    """Recognize formatting-only variants; never fuzzy-match or guess a hash."""
    candidate = value.strip().removeprefix("Evidence ID: ").strip()
    if candidate in allowed:
        return candidate
    if re.fullmatch(r"[0-9a-f]{64}", candidate) and f"ev_{candidate}" in allowed:
        return f"ev_{candidate}"
    return value


class ResearcherAgent:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        evidence = [
            EvidenceChunk.model_validate(item) for item in state.get("evidence", [])
        ]
        synthesis, metadata = self.provider.generate_structured(
            researcher_messages(
                state["question"], ResearchPlan.model_validate(state["plan"]), evidence
            ),
            ResearchSynthesis,
        )
        allowed = {item.id for item in evidence}
        synthesis = synthesis.model_copy(deep=True)
        for finding in synthesis.findings:
            finding.supporting_evidence_ids = [_canonical_evidence_id(value, allowed) for value in finding.supporting_evidence_ids]
            finding.conflicting_evidence_ids = [_canonical_evidence_id(value, allowed) for value in finding.conflicting_evidence_ids]
        referenced = {
            evidence_id
            for finding in synthesis.findings
            for evidence_id in [
                *finding.supporting_evidence_ids,
                *finding.conflicting_evidence_ids,
            ]
        }
        unknown = referenced - allowed
        if unknown:
            error = CitationError(
                "unknown_evidence_id",
                f"researcher referenced {len(unknown)} unknown evidence IDs",
            )
            error.metadata = metadata
            raise error
        return {
            "synthesis": synthesis.model_dump(mode="json"),
            "next_stage": "critic",
            "provider_metrics": [
                *state.get("provider_metrics", []),
                metadata.model_dump(mode="json"),
            ],
        }
