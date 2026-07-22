from dataclasses import dataclass

from app.agents.retriever import RetrieverAgent
from app.domain.documents import EvidenceChunk
from app.domain.providers import ProviderMetadata
from app.domain.research import Critique, ResearchPlan
from app.retrieval.contracts import RetrievalBatch
from app.workflow.state import initial_state


def evidence(evidence_id: str, score: float) -> EvidenceChunk:
    return EvidenceChunk(
        id=evidence_id,
        document_id="doc",
        filename="source.txt",
        page_number=None,
        chunk_index=0,
        content=f"content for {evidence_id}",
        content_sha256=f"hash-{evidence_id}",
        score=score,
    )


@dataclass
class RetrievalCall:
    question: str
    expansions: list[str]
    document_ids: list[str]
    top_k: int


class FakeRetriever:
    def __init__(self, batch: RetrievalBatch) -> None:
        self.batch = batch
        self.calls: list[RetrievalCall] = []

    def retrieve(
        self,
        question: str,
        expansions: list[str],
        document_ids: list[str],
        top_k: int,
    ) -> RetrievalBatch:
        self.calls.append(RetrievalCall(question, expansions, document_ids, top_k))
        return self.batch


def test_retriever_combines_queries_once_and_keeps_highest_scored_evidence() -> None:
    """Removing a query, another retrieval call, or score-aware merge must fail."""
    plan = ResearchPlan(
        objective="compare",
        subquestions=["what agrees?"],
        search_queries=["agreement", "conflict"],
        completion_criteria=["cite every finding"],
    )
    retriever = FakeRetriever(
        RetrievalBatch(
            evidence=[evidence("duplicate", 0.7), evidence("alpha", 0.8)],
            provider_metrics=[
                ProviderMetadata(provider="fake", model="fake-hash-16", latency_ms=1)
            ],
        )
    )
    state = initial_state("run", "question", ["doc"])
    state.update(
        plan=plan.model_dump(mode="json"),
        critique=Critique(
            sufficient=False,
            reason="gap",
            evidence_gaps=["missing"],
            follow_up_queries=["follow up"],
        ).model_dump(mode="json"),
        evidence=[
            evidence("duplicate", 0.9).model_dump(mode="json"),
            evidence("zeta", 0.8).model_dump(mode="json"),
        ],
        provider_metrics=[
            ProviderMetadata(provider="planner", model="chat", latency_ms=1).model_dump(
                mode="json"
            )
        ],
    )

    update = RetrieverAgent(retriever, top_k=6)(state)

    assert len(retriever.calls) == 1
    assert retriever.calls[0] == RetrievalCall(
        question="question",
        expansions=["agreement", "conflict", "follow up"],
        document_ids=["doc"],
        top_k=6,
    )
    assert [(item["id"], item["score"]) for item in update["evidence"]] == [
        ("duplicate", 0.9),
        ("alpha", 0.8),
        ("zeta", 0.8),
    ]
    assert update["next_stage"] == "researcher"
    assert [item["provider"] for item in update["provider_metrics"]] == [
        "planner",
        "fake",
    ]
