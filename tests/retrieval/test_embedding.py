from math import isclose
from typing import Sequence

import pytest

from app.domain.documents import EvidenceChunk
from app.domain.errors import RetrievalError
from app.domain.providers import ProviderMetadata
from app.providers.fake import DeterministicEmbeddingProvider
from app.retrieval.embedding import embed_chunks


def _chunks(count: int) -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            id=f"chunk-{index}",
            document_id="doc-1",
            filename="notes.txt",
            page_number=None,
            chunk_index=index,
            content=f"content {index}",
            content_sha256=f"digest-{index}",
        )
        for index in range(count)
    ]


class RecordingEmbeddingProvider:
    model_name = "recording-model"

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.batch_sizes: list[int] = []
        self.metadata: list[ProviderMetadata] = []

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], ProviderMetadata]:
        self.batch_sizes.append(len(texts))
        metadata = ProviderMetadata(
            provider="recording",
            model=self.model_name,
            latency_ms=len(self.batch_sizes),
        )
        self.metadata.append(metadata)
        vectors = self.vectors if self.vectors is not None else [[3.0, 4.0] for _ in texts]
        return vectors, metadata


def test_embed_chunks_preserves_order_model_and_metadata() -> None:
    chunks = _chunks(3)
    provider = DeterministicEmbeddingProvider(dimensions=16)

    embedded, metrics = embed_chunks(chunks, provider, batch_size=2)

    assert [item[0].id for item in embedded] == [chunk.id for chunk in chunks]
    assert [item[0].content for item in embedded] == [chunk.content for chunk in chunks]
    assert all(item[0].embedding_model == "fake-hash-16" for item in embedded)
    assert all(len(item[1]) == 16 for item in embedded)
    assert [item.model for item in metrics] == ["fake-hash-16", "fake-hash-16"]


def test_embed_chunks_batches_sixty_five_chunks_as_32_32_1() -> None:
    provider = RecordingEmbeddingProvider()

    embedded, metrics = embed_chunks(_chunks(65), provider, batch_size=32)

    assert provider.batch_sizes == [32, 32, 1]
    assert metrics == provider.metadata
    assert len(embedded) == 65


def test_embed_chunks_normalizes_each_vector() -> None:
    embedded, _ = embed_chunks(
        _chunks(2), RecordingEmbeddingProvider(), batch_size=2
    )

    assert embedded[0][1] == pytest.approx([0.6, 0.8])
    assert all(isclose(sum(value * value for value in vector), 1.0) for _, vector in embedded)


def test_embed_chunks_rejects_count_mismatch() -> None:
    provider = RecordingEmbeddingProvider(vectors=[[1.0, 0.0]])

    with pytest.raises(RetrievalError) as raised:
        embed_chunks(_chunks(2), provider, batch_size=32)

    assert raised.value.code == "embedding_count_mismatch"


def test_embed_chunks_rejects_dimension_mismatch() -> None:
    provider = RecordingEmbeddingProvider(vectors=[[1.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(RetrievalError) as raised:
        embed_chunks(_chunks(2), provider, batch_size=32)

    assert raised.value.code == "embedding_dimension_mismatch"


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")]])
def test_embed_chunks_rejects_empty_or_non_finite_vectors(vector: list[float]) -> None:
    provider = RecordingEmbeddingProvider(vectors=[vector])

    with pytest.raises(RetrievalError) as raised:
        embed_chunks(_chunks(1), provider, batch_size=32)

    assert raised.value.code == "invalid_embedding"


def test_embed_chunks_rejects_zero_vectors() -> None:
    provider = RecordingEmbeddingProvider(vectors=[[0.0, 0.0]])

    with pytest.raises(RetrievalError) as raised:
        embed_chunks(_chunks(1), provider, batch_size=32)

    assert raised.value.code == "zero_embedding"


@pytest.mark.parametrize("batch_size", [0, -1])
def test_embed_chunks_rejects_non_positive_batch_size(batch_size: int) -> None:
    with pytest.raises(RetrievalError) as raised:
        embed_chunks(_chunks(1), RecordingEmbeddingProvider(), batch_size=batch_size)

    assert raised.value.code == "invalid_embedding_batch_size"
