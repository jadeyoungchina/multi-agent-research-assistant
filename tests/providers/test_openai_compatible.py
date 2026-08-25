from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIError

from app.domain.errors import ProviderError
from app.domain.providers import ChatMessage
from app.domain.research import Critique
from app.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)


VALID_CRITIQUE = (
    '{"sufficient":true,"reason":"ok","evidence_gaps":[],"follow_up_queries":[]}'
)


def chat_response(content: Any, *, model: str = "qwen3.7-flash") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model=model,
    )


class SequencedCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = iter(outcomes)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def chat_client(completions: SequencedCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_structured_completion_is_validated() -> None:
    completions = SequencedCompletions([chat_response(VALID_CRITIQUE)])
    provider = OpenAICompatibleChatProvider(
        client=chat_client(completions),
        provider_name="dashscope",
        model="qwen3.7-flash",
        max_retries=0,
    )

    result, metadata = provider.generate_structured(
        [ChatMessage(role="user", content="review")], Critique
    )

    assert result.sufficient is True
    assert metadata.provider == "dashscope"
    assert metadata.model == "qwen3.7-flash"
    assert metadata.usage.total_tokens == 15
    assert completions.calls == [
        {
            "model": "qwen3.7-flash",
            "messages": [{"role": "user", "content": "review"}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ]


def test_chat_retries_twice_before_success(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr("app.providers.openai_compatible.sleep", delays.append)
    completions = SequencedCompletions(
        [TimeoutError("first"), TimeoutError("second"), chat_response("answer")]
    )
    provider = OpenAICompatibleChatProvider(
        client=chat_client(completions),
        provider_name="openai",
        model="gpt-4o-mini",
        max_retries=2,
    )

    result, metadata = provider.generate([ChatMessage(role="user", content="question")])

    assert result == "answer"
    assert metadata.retries == 2
    assert delays == [0.25, 0.5]


def test_structured_retry_appends_schema_repair_without_mutating_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.providers.openai_compatible.sleep", lambda _: None)
    completions = SequencedCompletions(
        [chat_response("not-json"), chat_response(VALID_CRITIQUE)]
    )
    provider = OpenAICompatibleChatProvider(
        client=chat_client(completions),
        provider_name="dashscope",
        model="qwen3.7-flash",
        max_retries=1,
    )
    messages = [ChatMessage(role="system", content="be exact"), ChatMessage(role="user", content="review")]

    result, metadata = provider.generate_structured(messages, Critique)

    assert result.sufficient is True
    assert metadata.retries == 1
    assert messages == [
        ChatMessage(role="system", content="be exact"),
        ChatMessage(role="user", content="review"),
    ]
    retry_messages = completions.calls[1]["messages"]
    assert retry_messages[:2] == [
        {"role": "system", "content": "be exact"},
        {"role": "user", "content": "review"},
    ]
    assert retry_messages[2]["role"] == "user"
    assert "Critique" in retry_messages[2]["content"]
    assert "evidence_gaps" in retry_messages[2]["content"]


def test_invalid_structured_response_has_safe_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.providers.openai_compatible.sleep", lambda _: None)
    secret = "sk-sensitive-value"
    completions = SequencedCompletions(
        [chat_response(f"invalid {secret}"), chat_response(f"still invalid {secret}")]
    )
    provider = OpenAICompatibleChatProvider(
        client=chat_client(completions),
        provider_name="openai",
        model="gpt-4o-mini",
        max_retries=1,
    )

    with pytest.raises(ProviderError) as raised:
        provider.generate_structured([ChatMessage(role="user", content="review")], Critique)

    assert raised.value.code == "provider_invalid_response"
    assert secret not in str(raised.value)
    assert "still invalid" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (TimeoutError("request timed out with sk-sensitive-value"), "provider_timeout"),
        (
            APIError(
                "upstream rejected sk-sensitive-value",
                request=SimpleNamespace(),
                body=None,
            ),
            "provider_request_failed",
        ),
    ],
)
def test_chat_failures_are_classified_without_leaking_details(
    error: Exception, expected_code: str
) -> None:
    provider = OpenAICompatibleChatProvider(
        client=chat_client(SequencedCompletions([error])),
        provider_name="openai",
        model="gpt-4o-mini",
        max_retries=0,
    )

    with pytest.raises(ProviderError) as raised:
        provider.generate([ChatMessage(role="user", content="question")])

    assert raised.value.code == expected_code
    assert "sk-sensitive-value" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[], usage=None, model="gpt-4o-mini"),
        chat_response(None, model="gpt-4o-mini"),
    ],
)
def test_malformed_chat_response_is_normalized(response: SimpleNamespace) -> None:
    provider = OpenAICompatibleChatProvider(
        client=chat_client(SequencedCompletions([response])),
        provider_name="openai",
        model="gpt-4o-mini",
        max_retries=0,
    )

    with pytest.raises(ProviderError) as raised:
        provider.generate([ChatMessage(role="user", content="question")])

    assert raised.value.code == "provider_invalid_response"


def test_malformed_unstructured_metadata_retries_as_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.providers.openai_compatible.sleep", lambda _: None)
    malformed = chat_response("answer", model=None)
    provider = OpenAICompatibleChatProvider(
        client=chat_client(SequencedCompletions([malformed, malformed])),
        provider_name="openai",
        model="gpt-4o-mini",
        max_retries=1,
    )

    with pytest.raises(ProviderError) as raised:
        provider.generate([ChatMessage(role="user", content="question")])

    assert raised.value.code == "provider_invalid_response"


class SequencedEmbeddings:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = iter(outcomes)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def embedding_response(*, data: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        data=data
        if data is not None
        else [
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ],
        usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
        model="text-embedding-v4",
    )


def test_embedding_vectors_are_ordered_with_metadata() -> None:
    embeddings = SequencedEmbeddings([embedding_response()])
    provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(embeddings=embeddings),
        provider_name="dashscope",
        model="text-embedding-v4",
        max_retries=0,
    )

    vectors, metadata = provider.embed_documents(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert provider.model_name == "text-embedding-v4"
    assert metadata.provider == "dashscope"
    assert metadata.model == "text-embedding-v4"
    assert metadata.latency_ms >= 0
    assert metadata.retries == 0
    assert metadata.usage.prompt_tokens == 7
    assert metadata.usage.total_tokens == 7
    assert embeddings.calls == [
        {"model": "text-embedding-v4", "input": ["first", "second"]}
    ]


def test_embedding_retries_exponentially(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr("app.providers.openai_compatible.sleep", delays.append)
    embeddings = SequencedEmbeddings(
        [RuntimeError("one"), RuntimeError("two"), embedding_response()]
    )
    provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(embeddings=embeddings),
        provider_name="openai",
        model="text-embedding-3-small",
        max_retries=2,
    )

    vectors, metadata = provider.embed_documents(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert metadata.retries == 2
    assert delays == [0.25, 0.5]


@pytest.mark.parametrize(
    "data",
    [
        [SimpleNamespace(index=0, embedding=[0.1, 0.2])],
        [
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            SimpleNamespace(index=0, embedding=[0.3, 0.4]),
        ],
        [
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
            SimpleNamespace(index=2, embedding=[0.3, 0.4]),
        ],
    ],
)
def test_embedding_response_must_match_requested_vector_count_and_indexes(
    data: list[SimpleNamespace],
) -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(embeddings=SequencedEmbeddings([embedding_response(data=data)])),
        provider_name="dashscope",
        model="text-embedding-v4",
        max_retries=0,
    )

    with pytest.raises(ProviderError) as raised:
        provider.embed_documents(["first", "second"])

    assert raised.value.code == "provider_embedding_failed"


def test_embedding_failure_is_normalized_without_leaking_details() -> None:
    secret = "sk-sensitive-value"
    provider = OpenAICompatibleEmbeddingProvider(
        client=SimpleNamespace(
            embeddings=SequencedEmbeddings([RuntimeError(f"response and {secret}")])
        ),
        provider_name="openai",
        model="text-embedding-3-small",
        max_retries=0,
    )

    with pytest.raises(ProviderError) as raised:
        provider.embed_query("query")

    assert raised.value.code == "provider_embedding_failed"
    assert secret not in str(raised.value)
    assert "response" not in str(raised.value)
    assert raised.value.__cause__ is None
