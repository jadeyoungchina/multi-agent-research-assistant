from collections.abc import Sequence

import pytest

from app.domain.documents import EvidenceChunk
from app.domain.errors import ProviderError, RetrievalError
from app.domain.providers import ProviderMetadata, TokenUsage
from app.retrieval.retriever import EvidenceRetriever


def metadata(model: str) -> ProviderMetadata:
    return ProviderMetadata(
        provider="fake",
        model=model,
        latency_ms=3,
        retries=1,
        usage=TokenUsage(prompt_tokens=2, total_tokens=2),
    )


def chunk(evidence_id: str, *, page_number: int | None = 2) -> EvidenceChunk:
    return EvidenceChunk(
        id=evidence_id,
        document_id="doc1",
        filename="source.pdf",
        page_number=page_number,
        chunk_index=4,
        content=f"content for {evidence_id}",
        content_sha256=f"sha-{evidence_id}",
        embedding_model="embedding-model",
        score=0.8,
    )


class FakeEmbeddings:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> tuple[list[float], ProviderMetadata]:
        self.queries.append(text)
        return [float(len(self.queries)), 0.0], metadata(f"model-{len(self.queries)}")


class FailingEmbeddings(FakeEmbeddings):
    def embed_query(self, text: str) -> tuple[list[float], ProviderMetadata]:
        raise ProviderError("provider_embedding_failed", "embedding request failed")


class FakeIndex:
    def __init__(self, results: list[list[EvidenceChunk]] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[tuple[list[float], list[str], int, float]] = []

    def search(
        self,
        query_vector: Sequence[float],
        document_ids: Sequence[str],
        limit: int,
        min_score: float,
    ) -> list[EvidenceChunk]:
        self.calls.append((list(query_vector), list(document_ids), limit, min_score))
        return self.results.pop(0) if self.results else []


class FailingIndex(FakeIndex):
    def __init__(self, error: RetrievalError) -> None:
        super().__init__()
        self.error = error

    def search(
        self,
        query_vector: Sequence[float],
        document_ids: Sequence[str],
        limit: int,
        min_score: float,
    ) -> list[EvidenceChunk]:
        raise self.error


def retriever(embeddings: FakeEmbeddings, index: FakeIndex) -> EvidenceRetriever:
    return EvidenceRetriever(
        embeddings,
        index,
        candidate_k=20,
        rrf_k=60,
        min_similarity=0.15,
        max_query_expansions=4,
    )


def test_retriever_embeds_each_unique_query_once_and_preserves_metrics() -> None:
    embeddings = FakeEmbeddings()
    index = FakeIndex([[chunk("a")], [chunk("a"), chunk("b")]])

    batch = retriever(embeddings, index).retrieve(
        "Question", ["query", "QUERY"], ["doc1"], top_k=6
    )

    assert embeddings.queries == ["Question", "query"]
    assert [item.model for item in batch.provider_metrics] == ["model-1", "model-2"]
    assert [item.id for item in batch.evidence] == ["a", "b"]


def test_retriever_calls_index_with_candidate_limit_and_minimum_similarity() -> None:
    embeddings = FakeEmbeddings()
    index = FakeIndex()

    retriever(embeddings, index).retrieve("Question", [], ["doc1", "doc2"], top_k=2)

    assert index.calls == [([1.0, 0.0], ["doc1", "doc2"], 20, 0.15)]


def test_retriever_rejects_an_empty_document_selection_before_embedding() -> None:
    embeddings = FakeEmbeddings()

    with pytest.raises(RetrievalError, match="at least one document") as error:
        retriever(embeddings, FakeIndex()).retrieve("Question", [], [], top_k=2)

    assert error.value.code == "no_documents_selected"
    assert embeddings.queries == []


def test_retriever_returns_metrics_when_no_candidates_are_found() -> None:
    embeddings = FakeEmbeddings()

    batch = retriever(embeddings, FakeIndex()).retrieve(
        "Question", ["query"], ["doc1"], top_k=2
    )

    assert batch.evidence == []
    assert [item.model for item in batch.provider_metrics] == ["model-1", "model-2"]


def test_retriever_propagates_embedding_provider_errors() -> None:
    with pytest.raises(ProviderError, match="embedding request failed") as error:
        retriever(FailingEmbeddings(), FakeIndex()).retrieve(
            "Question", [], ["doc1"], top_k=2
        )

    assert error.value.code == "provider_embedding_failed"


def test_retriever_propagates_safe_retrieval_errors() -> None:
    expected = RetrievalError("invalid_query_embedding", "query embedding is invalid")

    with pytest.raises(RetrievalError) as error:
        retriever(FakeEmbeddings(), FailingIndex(expected)).retrieve(
            "Question", [], ["doc1"], top_k=2
        )

    assert error.value is expected
    assert str(error.value) == "query embedding is invalid"


def test_retriever_preserves_filename_page_and_chunk_metadata() -> None:
    original = chunk("a", page_number=8)

    batch = retriever(FakeEmbeddings(), FakeIndex([[original]])).retrieve(
        "Question", [], ["doc1"], top_k=1
    )

    assert batch.evidence[0].model_dump(exclude={"score"}) == original.model_dump(
        exclude={"score"}
    )
    assert batch.evidence[0].filename == "source.pdf"
    assert batch.evidence[0].page_number == 8
    assert batch.evidence[0].chunk_index == 4
