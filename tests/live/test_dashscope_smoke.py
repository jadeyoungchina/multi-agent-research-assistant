import os

import pytest

from app.config import get_settings
from app.domain.providers import ChatMessage
from app.providers.factory import build_chat_provider


@pytest.mark.live
def test_dashscope_qwen_smoke() -> None:
    if os.getenv("RUN_LIVE_PROVIDER_TESTS") != "1":
        pytest.skip("live provider tests are opt-in")

    settings = get_settings()
    if settings.dashscope_api_key is None:
        pytest.skip("DASHSCOPE_API_KEY is not configured")

    provider = build_chat_provider(settings)
    text, metadata = provider.generate(
        [ChatMessage(role="user", content="Reply with exactly: pong")]
    )

    assert text.strip().casefold() == "pong"
    assert metadata.model == "qwen3.7-flash"
