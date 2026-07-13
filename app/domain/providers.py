from typing import Literal, Protocol, Sequence, TypeVar

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ProviderMetadata(BaseModel):
    provider: str
    model: str
    latency_ms: int
    retries: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)


T = TypeVar("T", bound=BaseModel)


class ChatProvider(Protocol):
    def generate(self, messages: Sequence[ChatMessage]) -> tuple[str, ProviderMetadata]: ...

    def generate_structured(
        self, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, ProviderMetadata]: ...


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], ProviderMetadata]: ...

    def embed_query(self, text: str) -> tuple[list[float], ProviderMetadata]: ...
