from typing import Any

from openai import OpenAI
from pydantic import SecretStr

from app.config import Settings
from app.domain.errors import ConfigurationError
from app.domain.providers import ChatProvider, EmbeddingProvider
from app.providers.fake import DeterministicEmbeddingProvider, build_demo_fake_provider
from app.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)


def _secret_value(secret: SecretStr | None, name: str) -> str:
    value = secret.get_secret_value() if secret is not None else ""
    if not value:
        raise ConfigurationError("missing_api_key", f"{name} is not configured")
    return value


def _client(api_key: str, base_url: str, timeout: float) -> Any:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def build_chat_provider(settings: Settings) -> ChatProvider:
    if settings.chat_provider == "fake":
        return build_demo_fake_provider()

    if settings.chat_provider == "dashscope":
        client = _client(
            _secret_value(settings.dashscope_api_key, "DASHSCOPE_API_KEY"),
            settings.dashscope_base_url,
            settings.provider_timeout_seconds,
        )
        return OpenAICompatibleChatProvider(
            client=client,
            provider_name="dashscope",
            model=settings.dashscope_chat_model,
            max_retries=settings.provider_max_retries,
        )

    client = _client(
        _secret_value(settings.openai_api_key, "OPENAI_API_KEY"),
        settings.openai_base_url,
        settings.provider_timeout_seconds,
    )
    return OpenAICompatibleChatProvider(
        client=client,
        provider_name="openai",
        model=settings.openai_chat_model,
        max_retries=settings.provider_max_retries,
    )


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "fake":
        return DeterministicEmbeddingProvider()

    if settings.embedding_provider == "dashscope":
        client = _client(
            _secret_value(settings.dashscope_api_key, "DASHSCOPE_API_KEY"),
            settings.dashscope_base_url,
            settings.provider_timeout_seconds,
        )
        return OpenAICompatibleEmbeddingProvider(
            client=client,
            provider_name="dashscope",
            model=settings.dashscope_embedding_model,
            max_retries=settings.provider_max_retries,
        )

    client = _client(
        _secret_value(settings.openai_api_key, "OPENAI_API_KEY"),
        settings.openai_base_url,
        settings.provider_timeout_seconds,
    )
    return OpenAICompatibleEmbeddingProvider(
        client=client,
        provider_name="openai",
        model=settings.openai_embedding_model,
        max_retries=settings.provider_max_retries,
    )
