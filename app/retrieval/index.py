from collections.abc import Iterable, Sequence

import numpy as np

from app.domain.errors import RetrievalError
from app.domain.documents import EvidenceChunk


class LocalVectorIndex:
    def __init__(self, repository) -> None:
        self.repository = repository

    @staticmethod
    def _normalized_vector(
        value: Sequence[float], error_code: str, error_message: str
    ) -> np.ndarray:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RetrievalError(error_code, error_message) from exc
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise RetrievalError(error_code, error_message)
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise RetrievalError(error_code, error_message)
        return vector / norm

    def search(
        self,
        query_vector: Sequence[float],
        document_ids: Iterable[str],
        limit: int,
        min_score: float,
    ) -> list[EvidenceChunk]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise RetrievalError(
                "invalid_result_limit", "limit must be a positive integer"
            )
        rows = self.repository.list_chunks(list(document_ids))
        if not rows:
            return []
        query = self._normalized_vector(
            query_vector,
            "invalid_query_embedding",
            "query embedding must be a non-zero vector",
        )
        ranked: list[EvidenceChunk] = []
        for chunk, vector in rows:
            candidate = self._normalized_vector(
                vector,
                "invalid_stored_embedding",
                f"stored embedding {chunk.id} is not a finite non-zero vector",
            )
            if candidate.shape != query.shape:
                raise RetrievalError(
                    "embedding_dimension_mismatch",
                    "query and document embedding dimensions differ",
                )
            score = float(np.dot(query, candidate))
            if score >= min_score:
                ranked.append(chunk.model_copy(update={"score": score}))
        return sorted(ranked, key=lambda item: (-float(item.score), item.id))[:limit]
