from typing import Any

from openai import OpenAI
from pydantic import SecretStr

from app.config import Settings
from app.domain.errors import ConfigurationError
from app.domain.providers import ChatProvider, EmbeddingProvider
from app.evaluation.budget import BudgetError, BudgetLedger
from app.providers.fake import DeterministicEmbeddingProvider, build_demo_fake_provider
from app.providers.openai_compatible import OpenAICompatibleChatProvider, OpenAICompatibleEmbeddingProvider


def _secret_value(secret: SecretStr | None, name: str) -> str:
    value = secret.get_secret_value() if secret is not None else ""
    if not value:
        raise ConfigurationError("missing_api_key", f"{name} is not configured")
    return value


def _client(api_key: str, base_url: str, timeout: float) -> Any:
    # The application owns retries, including every billable physical attempt.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


def _provider_client(settings: Settings, provider: str, ledger: BudgetLedger | None):
    endpoint = getattr(settings, f"{provider}_base_url")
    allowed = {
        "dashscope": {"https://dashscope.aliyuncs.com/compatible-mode/v1"},
        "deepseek": {"https://api.deepseek.com", "https://api.deepseek.com/v1"},
    }
    if ledger is not None and endpoint.rstrip("/") not in allowed.get(provider, set()):
        raise BudgetError("budget_unpriced_endpoint", "endpoint does not match the verified price region")
    return _client(_secret_value(getattr(settings, f"{provider}_api_key"), f"{provider.upper()}_API_KEY"),
                   endpoint, settings.provider_timeout_seconds)


def build_chat_provider(settings: Settings, *, ledger: BudgetLedger | None = None,
                        role: str = "chat") -> ChatProvider:
    provider = settings.chat_provider
    if provider == "fake":
        return build_demo_fake_provider()
    return OpenAICompatibleChatProvider(
        client=_provider_client(settings, provider, ledger), provider_name=provider,
        model=getattr(settings, f"{provider}_chat_model"), max_retries=settings.provider_max_retries,
        max_output_tokens=settings.provider_max_output_tokens, ledger=ledger, role=role,
        structured_output=settings.provider_structured_output,
    )


def build_embedding_provider(settings: Settings, *, ledger: BudgetLedger | None = None,
                             role: str = "embedding") -> EmbeddingProvider:
    provider = settings.embedding_provider
    if provider == "fake":
        return DeterministicEmbeddingProvider()
    return OpenAICompatibleEmbeddingProvider(
        client=_provider_client(settings, provider, ledger), provider_name=provider,
        model=getattr(settings, f"{provider}_embedding_model"),
        max_retries=settings.provider_max_retries, ledger=ledger, role=role,
    )
