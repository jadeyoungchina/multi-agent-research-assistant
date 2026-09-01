from collections.abc import Sequence as RuntimeSequence
from math import isfinite
from numbers import Real
from time import perf_counter, sleep
from typing import Any, Sequence, TypeVar

from openai import APIError, APITimeoutError
from pydantic import BaseModel, ValidationError

from app.domain.errors import ProviderError
from app.domain.providers import ChatMessage, ProviderMetadata, TokenUsage

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleChatProvider:
    def __init__(
        self,
        client: Any,
        provider_name: str,
        model: str,
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.provider_name = provider_name
        self.model = model
        self.max_retries = max_retries

    def _request(
        self, messages: Sequence[ChatMessage], *, json_mode: bool
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**kwargs)

    def generate(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[str, ProviderMetadata]:
        content, metadata = self._complete_with_retries(messages, None)
        return content, metadata

    def generate_structured(
        self, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, ProviderMetadata]:
        content, metadata = self._complete_with_retries(messages, schema)
        return schema.model_validate_json(content), metadata

    def _complete_with_retries(
        self, messages: Sequence[ChatMessage], schema: type[BaseModel] | None
    ) -> tuple[str, ProviderMetadata]:
        started = perf_counter()
        request_messages = list(messages)
        schema_instruction: ChatMessage | None = None
        if schema is not None:
            schema_instruction = ChatMessage(
                role="user",
                content=(
                    "Return only a JSON object matching the "
                    f"{schema.__name__} Pydantic JSON schema: "
                    f"{schema.model_json_schema()}."
                ),
            )
            request_messages.append(schema_instruction)
        failure_code = "provider_request_failed"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._request(request_messages, json_mode=schema is not None)
                content = self._chat_content(response)
                if schema is not None:
                    schema.model_validate_json(content)
                return content, self._metadata(response, started, attempt)
            except ValidationError as exc:
                last_error = exc
                failure_code = "provider_invalid_response"
                if attempt < self.max_retries:
                    if schema_instruction is not None:
                        request_messages = [*messages, schema_instruction]
                    sleep(0.25 * (2**attempt))
            except (APITimeoutError, TimeoutError) as exc:
                last_error = exc
                failure_code = "provider_timeout"
                if attempt < self.max_retries:
                    sleep(0.25 * (2**attempt))
            except APIError as exc:
                last_error = exc
                failure_code = "provider_request_failed"
                if attempt < self.max_retries:
                    sleep(0.25 * (2**attempt))
            except (AttributeError, IndexError, TypeError) as exc:
                last_error = exc
                failure_code = "provider_invalid_response"
                if attempt < self.max_retries:
                    sleep(0.25 * (2**attempt))
            except Exception as exc:
                last_error = exc
                failure_code = "provider_request_failed"
                if attempt < self.max_retries:
                    sleep(0.25 * (2**attempt))

        raise ProviderError(
            failure_code, f"provider request failed: {type(last_error).__name__}"
        ) from None

    @staticmethod
    def _chat_content(response: Any) -> str:
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise TypeError("chat response content must be text")
        return content

    def _metadata(
        self, response: Any, started: float, attempt: int
    ) -> ProviderMetadata:
        usage = getattr(response, "usage", None)
        return ProviderMetadata(
            provider=self.provider_name,
            model=getattr(response, "model", self.model),
            latency_ms=int((perf_counter() - started) * 1000),
            retries=attempt,
            usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            ),
        )


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        client: Any,
        provider_name: str,
        model: str,
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.provider_name = provider_name
        self._model = model
        self.max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], ProviderMetadata]:
        started = perf_counter()
        input_texts = list(texts)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self._model, input=input_texts
                )
                vectors = self._ordered_vectors(response, len(input_texts))
                usage = getattr(response, "usage", None)
                return vectors, ProviderMetadata(
                    provider=self.provider_name,
                    model=getattr(response, "model", self._model),
                    latency_ms=int((perf_counter() - started) * 1000),
                    retries=attempt,
                    usage=TokenUsage(
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        total_tokens=getattr(usage, "total_tokens", 0),
                    ),
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep(0.25 * (2**attempt))

        raise ProviderError(
            "provider_embedding_failed",
            f"embedding request failed: {type(last_error).__name__}",
        ) from None

    def embed_query(self, text: str) -> tuple[list[float], ProviderMetadata]:
        vectors, metadata = self.embed_documents([text])
        return vectors[0], metadata

    @staticmethod
    def _ordered_vectors(response: Any, expected_count: int) -> list[list[float]]:
        data = list(response.data)
        indexes = [item.index for item in data]
        if len(data) != expected_count or sorted(indexes) != list(range(expected_count)):
            raise ValueError("embedding response indexes do not match inputs")

        vectors: list[list[float]] = []
        dimensions: int | None = None
        for item in sorted(data, key=lambda item: item.index):
            raw_vector = item.embedding
            if (
                not isinstance(raw_vector, RuntimeSequence)
                or isinstance(raw_vector, (str, bytes))
                or not raw_vector
            ):
                raise ValueError("embedding must be a non-empty sequence")

            vector: list[float] = []
            for value in raw_vector:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not isfinite(float(value))
                ):
                    raise ValueError("embedding values must be finite real numbers")
                vector.append(float(value))

            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError("embedding dimensions must be consistent")
            vectors.append(vector)

        return vectors
