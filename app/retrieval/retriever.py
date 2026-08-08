# Concept-only from EU_Green_Compliance_FAQ_Bot.ipynb cells 21, 25, and 36.
# Source commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46.
# Corrected semantics: preserve source metadata, use reciprocal-rank scores, and ignore
# duplicate evidence IDs within each ranking.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt.

from app.domain.errors import RetrievalError
from app.domain.providers import EmbeddingProvider
from app.retrieval.contracts import RetrievalBatch
from app.retrieval.fusion import normalize_queries, reciprocal_rank_fuse
from app.retrieval.index import LocalVectorIndex


class EvidenceRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        index: LocalVectorIndex,
        candidate_k: int = 20,
        rrf_k: int = 60,
        min_similarity: float = 0.15,
        max_query_expansions: int = 4,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.index = index
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.min_similarity = min_similarity
        self.max_query_expansions = max_query_expansions

    def retrieve(
        self,
        question: str,
        expansions: list[str],
        document_ids: list[str],
        top_k: int,
    ) -> RetrievalBatch:
        if not document_ids:
            raise RetrievalError(
                "no_documents_selected", "at least one document is required"
            )

        queries = normalize_queries(question, expansions, self.max_query_expansions)
        ranked = []
        provider_metrics = []
        for query in queries:
            vector, metadata = self.embedding_provider.embed_query(query)
            provider_metrics.append(metadata)
            ranked.append(
                self.index.search(
                    vector,
                    document_ids,
                    self.candidate_k,
                    min_score=self.min_similarity,
                )
            )

        return RetrievalBatch(
            evidence=reciprocal_rank_fuse(ranked, self.rrf_k, top_k),
            provider_metrics=provider_metrics,
        )
