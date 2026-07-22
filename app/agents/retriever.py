# Concept-only from EU_Green_Compliance_FAQ_Bot.ipynb cells 21, 25, and 36.
# Source commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46.
# Changes: inject local retrieval, merge typed state evidence by stable ID, and preserve deterministic scores.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt.

from typing import Any

from app.domain.documents import EvidenceChunk
from app.domain.research import Critique, ResearchPlan
from app.retrieval.contracts import EvidenceRetrieverProtocol
from app.workflow.state import WorkflowState


def _score(chunk: EvidenceChunk) -> float:
    return chunk.score if chunk.score is not None else float("-inf")


class RetrieverAgent:
    def __init__(self, retriever: EvidenceRetrieverProtocol, top_k: int) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def __call__(self, state: WorkflowState) -> dict[str, Any]:
        plan = ResearchPlan.model_validate(state["plan"])
        critique = (
            Critique.model_validate(state["critique"])
            if state.get("critique") is not None
            else None
        )
        expansions = [
            *plan.search_queries,
            *(critique.follow_up_queries if critique is not None else []),
        ]
        batch = self.retriever.retrieve(
            state["question"], expansions, state["document_ids"], self.top_k
        )

        evidence_by_id: dict[str, EvidenceChunk] = {}
        for item in [
            *(EvidenceChunk.model_validate(value) for value in state.get("evidence", [])),
            *batch.evidence,
        ]:
            current = evidence_by_id.get(item.id)
            if current is None or _score(item) > _score(current):
                evidence_by_id[item.id] = item

        merged_evidence = sorted(
            evidence_by_id.values(), key=lambda item: (-_score(item), item.id)
        )
        return {
            "evidence": [item.model_dump(mode="json") for item in merged_evidence],
            "next_stage": "researcher",
            "provider_metrics": [
                *state.get("provider_metrics", []),
                *(item.model_dump(mode="json") for item in batch.provider_metrics),
            ],
        }
