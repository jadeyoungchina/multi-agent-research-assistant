import os
import sys
from types import SimpleNamespace

import pytest

from app.bootstrap import _close_target
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
    try:
        text, metadata = provider.generate(
            [ChatMessage(role="user", content="Reply with exactly: pong")]
        )

        assert text.strip().casefold() == "pong"
        assert metadata.model == "qwen3.7-flash"
    finally:
        target = _close_target(provider)
        if target is not None:
            target.close()


@pytest.mark.parametrize(
    "result",
    [
        RuntimeError("generation failed"),
        ("not pong", SimpleNamespace(model="qwen3.7-flash")),
    ],
    ids=["generation-error", "assertion-error"],
)
def test_dashscope_qwen_smoke_closes_client_when_check_fails(
    monkeypatch: pytest.MonkeyPatch, result: RuntimeError | tuple[str, object]
) -> None:
    closed: list[bool] = []

    class FailingProvider:
        client = SimpleNamespace(close=lambda: closed.append(True))

        def generate(self, messages: list[ChatMessage]) -> tuple[str, object]:
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setenv("RUN_LIVE_PROVIDER_TESTS", "1")
    monkeypatch.setattr(
        sys.modules[__name__],
        "get_settings",
        lambda: SimpleNamespace(dashscope_api_key=object()),
    )
    monkeypatch.setattr(
        sys.modules[__name__], "build_chat_provider", lambda _: FailingProvider()
    )

    with pytest.raises((RuntimeError, AssertionError)):
        test_dashscope_qwen_smoke()

    assert closed == [True]
