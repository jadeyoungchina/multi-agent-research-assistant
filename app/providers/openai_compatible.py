import json
from collections.abc import Sequence as RuntimeSequence
from math import isfinite
from numbers import Real
from time import perf_counter, sleep
from typing import Any, Sequence, TypeVar

from openai import APIError, APITimeoutError
from pydantic import BaseModel, ValidationError

from app.domain.errors import ProviderError
from app.domain.providers import ChatMessage, ProviderMetadata
from app.evaluation.budget import BudgetError, BudgetLedger
from app.providers.accounting import RequestAccounting

T = TypeVar("T", bound=BaseModel)


def _retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    return isinstance(error, (APIError, TimeoutError, ValidationError, AttributeError, IndexError, TypeError, ValueError))


def _failure_code(error: Exception, *, embedding: bool = False) -> str:
    if isinstance(error, (APITimeoutError, TimeoutError)):
        return "provider_timeout"
    if embedding:
        return "provider_embedding_failed"
    if isinstance(error, (ValidationError, AttributeError, IndexError, TypeError, ValueError)):
        return "provider_invalid_response"
    return "provider_request_failed"


class OpenAICompatibleChatProvider:
    def __init__(self, client: Any, provider_name: str, model: str, max_retries: int = 2,
                 *, max_output_tokens: int = 2048, ledger: BudgetLedger | None = None,
                 role: str = "chat") -> None:
        self.client, self.provider_name, self.model = client, provider_name, model
        self.max_retries, self.max_output_tokens = max_retries, max_output_tokens
        self.ledger, self.role = ledger, role
        # Explicit diagnostic opt-in for public evaluation corpora only. Receipts
        # contain generated text and safe schema errors, never headers or Settings.
        self.response_observer = None

    def _payload(self, messages: Sequence[ChatMessage], *, json_mode: bool) -> dict:
        payload: dict[str, Any] = {
            "model": self.model, "messages": [message.model_dump() for message in messages],
            "temperature": 0, "max_tokens": self.max_output_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.provider_name == "dashscope":
            payload["extra_body"] = {"enable_thinking": False}
        elif self.provider_name == "deepseek":
            payload["extra_body"] = {"thinking": {"type": "disabled"}}
        return payload

    def generate(self, messages: Sequence[ChatMessage]) -> tuple[str, ProviderMetadata]:
        return self._complete_with_retries(messages, None)

    def generate_structured(self, messages: Sequence[ChatMessage], schema: type[T]) -> tuple[T, ProviderMetadata]:
        content, metadata = self._complete_with_retries(messages, schema)
        return schema.model_validate_json(content), metadata

    def _complete_with_retries(self, messages: Sequence[ChatMessage], schema: type[BaseModel] | None):
        started = perf_counter()
        request_messages = list(messages)
        if schema is not None:
            request_messages.append(ChatMessage(role="user", content=(
                f"Return only a JSON object matching the {schema.__name__} Pydantic JSON schema: "
                f"{json.dumps(schema.model_json_schema())}."
            )))
        payload = self._payload(request_messages, json_mode=schema is not None)
        accounting = RequestAccounting(self.ledger, self.provider_name, self.model, "chat", self.role)
        failure_code = "provider_request_failed"
        for attempt in range(self.max_retries + 1):
            identifier = accounting.reserve(payload, self.max_output_tokens)
            received = False
            try:
                response = self.client.chat.completions.create(**payload)
                accounting.record(identifier, getattr(response, "usage", None))
                received = True
                content = response.choices[0].message.content
                if self.response_observer is not None:
                    issues = []
                    if schema is not None and isinstance(content, str):
                        try:
                            schema.model_validate_json(content)
                        except ValidationError as validation:
                            issues = [{"loc": list(item["loc"]), "type": item["type"]}
                                      for item in validation.errors(include_input=False, include_context=False, include_url=False)]
                    self.response_observer({
                        "attempt_id": identifier, "schema": schema.__name__ if schema else None,
                        "content": content if isinstance(content, str) else None,
                        "finish_reason": getattr(response.choices[0], "finish_reason", None),
                        "validation_issues": issues,
                    })
                if not isinstance(content, str):
                    raise TypeError("chat response content must be text")
                if getattr(response.choices[0], "finish_reason", None) == "length":
                    raise ProviderError("provider_output_truncated", "provider output reached its token limit")
                if schema is not None:
                    schema.model_validate_json(content)
                metadata = self._metadata(accounting, started, attempt, getattr(response, "model", self.model))
                accounting.annotate(identifier, "success")
                return content, metadata
            except BudgetError:
                raise
            except Exception as error:
                failure_code = error.code if isinstance(error, ProviderError) else _failure_code(error)
                if not received:
                    accounting.unknown(identifier, failure_code)
                accounting.annotate(identifier, failure_code)
                if isinstance(error, ProviderError) or not _retryable(error) or attempt == self.max_retries:
                    break
                sleep(0.25 * (2 ** attempt))
        raise ProviderError(failure_code, "provider request failed", metadata=self._metadata(
            accounting, started, attempt, self.model,
        )) from None

    def _metadata(self, accounting, started, attempt, model):
        return ProviderMetadata(
            provider=self.provider_name, model=model, latency_ms=int((perf_counter() - started) * 1000),
            retries=attempt, usage=accounting.usage, request_kind="chat", attempt_count=attempt + 1,
            attempt_ids=accounting.ids, usage_complete=accounting.complete,
        )


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, client: Any, provider_name: str, model: str, max_retries: int = 2,
                 *, ledger: BudgetLedger | None = None, role: str = "embedding") -> None:
        self.client, self.provider_name, self._model = client, provider_name, model
        self.max_retries, self.ledger, self.role = max_retries, ledger, role

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> tuple[list[list[float]], ProviderMetadata]:
        started = perf_counter()
        input_texts = list(texts)
        payload = {"model": self._model, "input": input_texts}
        accounting = RequestAccounting(self.ledger, self.provider_name, self._model, "embedding", self.role)
        failure_code = "provider_embedding_failed"
        for attempt in range(self.max_retries + 1):
            identifier = accounting.reserve(payload, 0)
            received = False
            try:
                response = self.client.embeddings.create(**payload)
                accounting.record(identifier, getattr(response, "usage", None))
                received = True
                vectors = self._ordered_vectors(response, len(input_texts))
                metadata = ProviderMetadata(
                    provider=self.provider_name, model=getattr(response, "model", self._model),
                    latency_ms=int((perf_counter() - started) * 1000), retries=attempt,
                    usage=accounting.usage, request_kind="embedding", attempt_count=attempt + 1,
                    attempt_ids=accounting.ids, usage_complete=accounting.complete,
                )
                accounting.annotate(identifier, "success")
                return vectors, metadata
            except BudgetError:
                raise
            except Exception as error:
                failure_code = _failure_code(error, embedding=True)
                if not received:
                    accounting.unknown(identifier, failure_code)
                accounting.annotate(identifier, failure_code)
                if not _retryable(error) or attempt == self.max_retries:
                    break
                sleep(0.25 * (2 ** attempt))
        metadata = ProviderMetadata(
            provider=self.provider_name, model=self._model, latency_ms=int((perf_counter() - started) * 1000),
            retries=attempt, usage=accounting.usage, request_kind="embedding", attempt_count=attempt + 1,
            attempt_ids=accounting.ids, usage_complete=accounting.complete,
        )
        raise ProviderError(failure_code, "provider embedding request failed", metadata=metadata) from None

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
            if not isinstance(raw_vector, RuntimeSequence) or isinstance(raw_vector, (str, bytes)) or not raw_vector:
                raise ValueError("embedding must be a non-empty sequence")
            vector: list[float] = []
            for value in raw_vector:
                if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
                    raise ValueError("embedding values must be finite real numbers")
                vector.append(float(value))
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError("embedding dimensions must be consistent")
            vectors.append(vector)
        return vectors
