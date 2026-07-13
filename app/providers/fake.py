from collections import deque
from hashlib import sha256
from math import sqrt
from time import perf_counter
from typing import Sequence, TypeVar

from pydantic import BaseModel

from app.domain.providers import ChatMessage, ProviderMetadata

T = TypeVar("T", bound=BaseModel)


class FakeChatProvider:
    def __init__(
        self,
        text_responses: Sequence[str] = (),
        structured_responses: Sequence[BaseModel] = (),
    ) -> None:
        self._text = deque(text_responses)
        self._structured = deque(structured_responses)

    def generate(self, messages: Sequence[ChatMessage]) -> tuple[str, ProviderMetadata]:
        started = perf_counter()
        if not self._text:
            raise AssertionError("no fake text response queued")
        value = self._text.popleft()
        return value, ProviderMetadata(
            provider="fake",
            model="fake-chat",
            latency_ms=int((perf_counter() - started) * 1000),
        )

    def generate_structured(
        self, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, ProviderMetadata]:
        started = perf_counter()
        if not self._structured:
            raise AssertionError(f"no fake response queued for {schema.__name__}")
        value = self._structured.popleft()
        if not isinstance(value, schema):
            raise AssertionError(f"expected {schema.__name__}, got {type(value).__name__}")
        return value, ProviderMetadata(
            provider="fake",
            model="fake-chat",
            latency_ms=int((perf_counter() - started) * 1000),
        )


class DeterministicEmbeddingProvider:
    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return f"fake-hash-{self._dimensions}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        encoded = text.casefold().encode("utf-8")
        for offset in range(0, len(encoded), 4):
            digest = sha256(encoded[offset : offset + 4]).digest()
            index = int.from_bytes(digest[:2], "big") % self._dimensions
            vector[index] += -1.0 if digest[2] & 1 else 1.0
        norm = sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], ProviderMetadata]:
        started = perf_counter()
        vectors = [self._embed(text) for text in texts]
        return vectors, ProviderMetadata(
            provider="fake",
            model=self.model_name,
            latency_ms=int((perf_counter() - started) * 1000),
        )

    def embed_query(self, text: str) -> tuple[list[float], ProviderMetadata]:
        started = perf_counter()
        return self._embed(text), ProviderMetadata(
            provider="fake",
            model=self.model_name,
            latency_ms=int((perf_counter() - started) * 1000),
        )
