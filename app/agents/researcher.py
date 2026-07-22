# Concept-only from multi_agent_collaboration_system.ipynb cells 6 and 11-21.
# Source commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46.
# Changes: inject provider, synthesize against typed local evidence, and reject ungrounded citations deterministically.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt.

from typing import Any

from app.agents.prompts import researcher_messages
from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError
from app.domain.providers import ChatProvider
from app.domain.research import ResearchPlan, ResearchSynthesis
from app.workflow.state import WorkflowState


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
            raise CitationError(
                "unknown_evidence_id",
                f"researcher referenced {len(unknown)} unknown evidence IDs",
            )
        return {
            "synthesis": synthesis.model_dump(mode="json"),
            "next_stage": "critic",
            "provider_metrics": [
                *state.get("provider_metrics", []),
                metadata.model_dump(mode="json"),
            ],
        }
