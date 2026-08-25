from typing import Any

import pytest

from app.config import Settings
from app.domain.errors import ConfigurationError
from app.providers.fake import DeterministicEmbeddingProvider, FakeChatProvider
from app.providers.factory import build_chat_provider, build_embedding_provider
from app.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_fake_providers_do_not_require_api_keys() -> None:
    configured = settings(chat_provider="fake", embedding_provider="fake")

    assert isinstance(build_chat_provider(configured), FakeChatProvider)
    assert isinstance(build_embedding_provider(configured), DeterministicEmbeddingProvider)


@pytest.mark.parametrize(
    (
        "builder",
        "provider_field",
        "provider_value",
        "key_field",
        "base_url_field",
        "model_field",
        "provider_type",
    ),
    [
        (
            build_chat_provider,
            "chat_provider",
            "dashscope",
            "dashscope_api_key",
            "dashscope_base_url",
            "dashscope_chat_model",
            OpenAICompatibleChatProvider,
        ),
        (
            build_chat_provider,
            "chat_provider",
            "openai",
            "openai_api_key",
            "openai_base_url",
            "openai_chat_model",
            OpenAICompatibleChatProvider,
        ),
        (
            build_embedding_provider,
            "embedding_provider",
            "dashscope",
            "dashscope_api_key",
            "dashscope_base_url",
            "dashscope_embedding_model",
            OpenAICompatibleEmbeddingProvider,
        ),
        (
            build_embedding_provider,
            "embedding_provider",
            "openai",
            "openai_api_key",
            "openai_base_url",
            "openai_embedding_model",
            OpenAICompatibleEmbeddingProvider,
        ),
    ],
)
def test_factory_passes_exact_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
    builder: Any,
    provider_field: str,
    provider_value: str,
    key_field: str,
    base_url_field: str,
    model_field: str,
    provider_type: type,
) -> None:
    clients: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "app.providers.factory.OpenAI",
        lambda **kwargs: clients.append(kwargs) or object(),
    )
    configured = settings(
        **{
            provider_field: provider_value,
            key_field: f"{provider_value}-secret",
            base_url_field: f"https://{provider_value}.example.test/v1",
            model_field: f"{provider_value}-model",
            "provider_timeout_seconds": 12.5,
            "provider_max_retries": 4,
        }
    )

    provider = builder(configured)

    assert isinstance(provider, provider_type)
    assert clients == [
        {
            "api_key": f"{provider_value}-secret",
            "base_url": f"https://{provider_value}.example.test/v1",
            "timeout": 12.5,
        }
    ]
    assert provider.provider_name == provider_value
    assert provider.model if isinstance(provider, OpenAICompatibleChatProvider) else provider.model_name
    if isinstance(provider, OpenAICompatibleChatProvider):
        assert provider.model == f"{provider_value}-model"
    else:
        assert provider.model_name == f"{provider_value}-model"
    assert provider.max_retries == 4


@pytest.mark.parametrize(
    ("builder", "configured_provider", "expected_name"),
    [
        (build_chat_provider, {"chat_provider": "dashscope"}, "DASHSCOPE_API_KEY"),
        (build_chat_provider, {"chat_provider": "openai"}, "OPENAI_API_KEY"),
        (
            build_embedding_provider,
            {"embedding_provider": "dashscope"},
            "DASHSCOPE_API_KEY",
        ),
        (
            build_embedding_provider,
            {"embedding_provider": "openai"},
            "OPENAI_API_KEY",
        ),
    ],
)
def test_factory_requires_only_selected_provider_secret(
    builder: Any, configured_provider: dict[str, str], expected_name: str
) -> None:
    configured = settings(**configured_provider)

    with pytest.raises(ConfigurationError) as raised:
        builder(configured)

    assert raised.value.code == "missing_api_key"
    assert expected_name in str(raised.value)


def test_chat_factory_does_not_require_embedding_provider_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.providers.factory.OpenAI", lambda **_: object())
    configured = settings(
        chat_provider="openai",
        embedding_provider="dashscope",
        openai_api_key="openai-secret",
        dashscope_api_key=None,
    )

    provider = build_chat_provider(configured)

    assert isinstance(provider, OpenAICompatibleChatProvider)
    assert provider.provider_name == "openai"


def test_embedding_factory_does_not_require_chat_provider_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.providers.factory.OpenAI", lambda **_: object())
    configured = settings(
        chat_provider="dashscope",
        embedding_provider="openai",
        dashscope_api_key=None,
        openai_api_key="openai-secret",
    )

    provider = build_embedding_provider(configured)

    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.provider_name == "openai"
