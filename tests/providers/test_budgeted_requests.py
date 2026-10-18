import json
from types import SimpleNamespace as NS

import httpx
from openai import AuthenticationError, RateLimitError
import pytest

from app.config import Settings
from app.domain.errors import ProviderError
from app.domain.providers import ChatMessage
from app.domain.research import Critique
from app.evaluation.budget import BudgetError, BudgetLedger
from app.providers.factory import build_chat_provider, build_embedding_provider
from app.providers.openai_compatible import OpenAICompatibleChatProvider, OpenAICompatibleEmbeddingProvider


VALID = '{"sufficient":true,"reason":"ok","evidence_gaps":[],"follow_up_queries":[]}'
MESSAGES = [ChatMessage(role="user", content="review")]


def response(content=VALID, *, usage=True, reason="stop"):
    return NS(model="deepseek-flash", choices=[NS(message=NS(content=content), finish_reason=reason)],
              usage=NS(prompt_tokens=100, completion_tokens=20, total_tokens=120) if usage else None)


class Calls:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []
    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = next(self.outcomes)
        if isinstance(item, Exception):
            raise item
        return item


def provider(tmp_path, outcomes, monkeypatch, **options):
    monkeypatch.setattr("app.providers.openai_compatible.sleep", lambda _: None)
    ledger = BudgetLedger(tmp_path / "ledger.db")
    calls = Calls(outcomes)
    chat = OpenAICompatibleChatProvider(NS(chat=NS(completions=calls)), "deepseek", "deepseek-flash",
                                        ledger=ledger, **options)
    return chat, calls, ledger


def test_invalid_json_and_success_both_count_before_validation(tmp_path, monkeypatch):
    chat, calls, ledger = provider(tmp_path, [response("not JSON"), response()], monkeypatch)
    answer, metadata = chat.generate_structured(MESSAGES, Critique)
    assert answer.sufficient
    assert metadata.usage.prompt_tokens == 200 and metadata.usage.completion_tokens == 40
    assert metadata.attempt_count == 2 and len(metadata.attempt_ids) == 2
    assert [r["outcome"] for r in ledger.summary()["attempts"]] == ["provider_invalid_response", "success"]
    for request in calls.requests:
        assert request["extra_body"] == {"thinking": {"type": "disabled"}}
        assert request["max_tokens"] == 2048


def test_final_failure_retains_known_usage_and_safe_metadata(tmp_path, monkeypatch):
    chat, _, ledger = provider(tmp_path, [response("bad"), response("bad")], monkeypatch, max_retries=1)
    with pytest.raises(ProviderError) as caught:
        chat.generate_structured(MESSAGES, Critique)
    assert caught.value.metadata.usage.total_tokens == 240
    assert all(row["state"] == "settled" for row in ledger.summary()["attempts"])
    assert "bad" not in str(caught.value)


def test_unknown_timeout_cost_is_not_cleared_after_success(tmp_path, monkeypatch):
    chat, _, ledger = provider(tmp_path, [TimeoutError(), response("ok")], monkeypatch)
    _, metadata = chat.generate(MESSAGES)
    assert not metadata.usage_complete
    assert metadata.usage.total_tokens == 120
    assert [row["state"] for row in ledger.summary()["attempts"]] == ["unknown", "settled"]
    assert ledger.summary()["reserved_or_unknown_cny"] != "0"


@pytest.mark.parametrize("status,error_type,expected_attempts", [(401, AuthenticationError, 1), (429, RateLimitError, 2)])
def test_retry_policy_uses_status_and_has_one_owner(tmp_path, monkeypatch, status, error_type, expected_attempts):
    error = error_type("refused", response=httpx.Response(status, request=httpx.Request("POST", "https://example.test")), body=None)
    chat, calls, _ = provider(tmp_path, [error, response("ok")], monkeypatch)
    if status == 401:
        with pytest.raises(ProviderError):
            chat.generate(MESSAGES)
    else:
        assert chat.generate(MESSAGES)[0] == "ok"
    assert len(calls.requests) == expected_attempts


def test_truncated_output_is_recorded_without_increasing_limit_or_retrying(tmp_path, monkeypatch):
    chat, calls, ledger = provider(tmp_path, [response("partial", reason="length")], monkeypatch)
    with pytest.raises(ProviderError) as caught:
        chat.generate(MESSAGES)
    assert caught.value.code == "provider_output_truncated"
    assert len(calls.requests) == 1
    assert ledger.summary()["attempts"][0]["state"] == "settled"


def test_missing_usage_stays_reserved_instead_of_reporting_free_call(tmp_path, monkeypatch):
    chat, _, ledger = provider(tmp_path, [response("ok", usage=False)], monkeypatch)
    _, metadata = chat.generate(MESSAGES)
    assert not metadata.usage_complete
    assert ledger.summary()["attempts"][0]["state"] == "unknown"


def test_budget_denial_precedes_network_and_is_not_retried(tmp_path, monkeypatch):
    chat, calls, _ = provider(tmp_path, [response()], monkeypatch)
    chat.ledger = BudgetLedger(tmp_path / "tiny.db", total_cny="0.0001", stop_cny="0.0001")
    with pytest.raises(BudgetError) as caught:
        chat.generate(MESSAGES)
    assert caught.value.code == "budget_exhausted" and not calls.requests


def test_unpriced_model_is_refused_before_network(tmp_path, monkeypatch):
    chat, calls, _ = provider(tmp_path, [response()], monkeypatch)
    chat.model = "unknown-model"
    with pytest.raises(BudgetError):
        chat.generate(MESSAGES)
    assert not calls.requests


def test_invalid_embedding_response_usage_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("app.providers.openai_compatible.sleep", lambda _: None)
    ledger = BudgetLedger(tmp_path / "ledger.db")
    responses = Calls([
        NS(data=[], model="text-embedding-v4", usage=NS(prompt_tokens=7)),
        NS(data=[NS(index=0, embedding=[0.1])], model="text-embedding-v4", usage=NS(prompt_tokens=7)),
    ])
    embedding = OpenAICompatibleEmbeddingProvider(NS(embeddings=responses), "dashscope", "text-embedding-v4", ledger=ledger)
    vectors, metadata = embedding.embed_query("hello")
    assert vectors == [0.1]
    assert metadata.usage.prompt_tokens == 14 and metadata.request_kind == "embedding"
    assert all(row["kind"] == "embedding" for row in ledger.summary()["attempts"])


def test_deepseek_and_embedding_use_independent_secrets_and_zero_sdk_retries(monkeypatch):
    clients = []
    monkeypatch.setattr("app.providers.factory.OpenAI", lambda **kwargs: clients.append(kwargs) or object())
    settings = Settings(_env_file=None, chat_provider="deepseek", embedding_provider="dashscope",
                        deepseek_api_key="deepseek-secret", dashscope_api_key="embedding-secret")
    chat = build_chat_provider(settings)
    embedding = build_embedding_provider(settings)
    assert chat.model == "deepseek-flash" and embedding.model_name == "text-embedding-v4"
    assert [item["api_key"] for item in clients] == ["deepseek-secret", "embedding-secret"]
    assert all(item["max_retries"] == 0 for item in clients)
    assert "deepseek-secret" not in repr(settings)


def test_wrong_region_endpoint_cannot_use_beijing_prices(tmp_path):
    settings = Settings(_env_file=None, dashscope_api_key="local-secret", dashscope_base_url="https://other.example/v1")
    with pytest.raises(BudgetError):
        build_chat_provider(settings, ledger=BudgetLedger(tmp_path / "ledger.db"))


def test_partial_usage_retains_known_input_without_releasing_reservation(tmp_path, monkeypatch):
    partial = response("ok")
    partial.usage = NS(prompt_tokens=123)
    chat, _, ledger = provider(tmp_path, [partial], monkeypatch)
    _, metadata = chat.generate(MESSAGES)
    assert metadata.usage.prompt_tokens == 123 and not metadata.usage_complete
    attempt = ledger.summary()["attempts"][0]
    assert attempt["input_tokens"] == 123 and attempt["output_tokens"] is None
    assert attempt["state"] == "unknown" and ledger.summary()["total_cost_cny"] is None


def test_observed_usage_survives_a_ledger_write_failure(tmp_path, monkeypatch):
    from app.providers.accounting import RequestAccounting
    ledger = BudgetLedger(tmp_path / "ledger.db")
    accounting = RequestAccounting(ledger, "deepseek", "deepseek-flash", "chat", "test")
    identifier = accounting.reserve({"model": "deepseek-flash", "messages": []}, 100)
    def fail(*args, **kwargs):
        raise OSError("simulated storage failure")
    monkeypatch.setattr(ledger, "settle", fail)
    with pytest.raises(OSError):
        accounting.record(identifier, NS(prompt_tokens=10, completion_tokens=20))
    assert accounting.usage.prompt_tokens == 10 and accounting.usage.completion_tokens == 20
    assert ledger.summary()["attempts"][0]["state"] == "reserved"
