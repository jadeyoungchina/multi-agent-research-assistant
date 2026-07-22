from typing import Protocol

from pydantic import BaseModel, Field

from app.domain.documents import EvidenceChunk
from app.domain.providers import ProviderMetadata


class RetrievalBatch(BaseModel):
    evidence: list[EvidenceChunk]
    provider_metrics: list[ProviderMetadata] = Field(default_factory=list)


class EvidenceRetrieverProtocol(Protocol):
    def retrieve(
        self,
        question: str,
        expansions: list[str],
        document_ids: list[str],
        top_k: int,
    ) -> RetrievalBatch: ...
