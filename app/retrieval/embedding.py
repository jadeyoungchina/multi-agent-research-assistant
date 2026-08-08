from math import isfinite, sqrt

from app.domain.documents import EvidenceChunk
from app.domain.errors import RetrievalError
from app.domain.providers import EmbeddingProvider, ProviderMetadata


def _normalize(vector: list[float]) -> list[float]:
    if not vector or not all(isfinite(value) for value in vector):
        raise RetrievalError(
            "invalid_embedding", "embedding must contain finite values"
        )
    norm = sqrt(sum(value * value for value in vector))
    if not isfinite(norm):
        raise RetrievalError(
            "invalid_embedding", "embedding must contain finite values"
        )
    if norm == 0:
        raise RetrievalError("zero_embedding", "embedding vector cannot be zero")
    return [value / norm for value in vector]


def embed_chunks(
    chunks: list[EvidenceChunk], provider: EmbeddingProvider, batch_size: int
) -> tuple[list[tuple[EvidenceChunk, list[float]]], list[ProviderMetadata]]:
    if batch_size <= 0:
        raise RetrievalError(
            "invalid_embedding_batch_size", "embedding batch size must be positive"
        )

    embedded: list[tuple[EvidenceChunk, list[float]]] = []
    metrics: list[ProviderMetadata] = []
    dimensions: int | None = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors, metadata = provider.embed_documents(
            [chunk.content for chunk in batch]
        )
        metrics.append(metadata)
        if len(vectors) != len(batch):
            raise RetrievalError(
                "embedding_count_mismatch",
                "provider returned the wrong number of vectors",
            )
        for chunk, vector in zip(batch, vectors):
            normalized = _normalize(vector)
            if dimensions is None:
                dimensions = len(normalized)
            elif len(normalized) != dimensions:
                raise RetrievalError(
                    "embedding_dimension_mismatch",
                    "embedding dimensions changed within one document",
                )
            embedded.append(
                (
                    chunk.model_copy(update={"embedding_model": provider.model_name}),
                    normalized,
                )
            )
    return embedded, metrics
