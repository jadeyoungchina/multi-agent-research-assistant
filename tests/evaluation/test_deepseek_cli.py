from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil

import pytest

from app.evaluation.budget import BudgetError, BudgetLedger, PRICES
from app.evaluation.deepseek_cli import BudgetBridge, load_price, response_usage, run_task


PRICE = PRICES[("deepseek", "deepseek-flash")]


def payload():
    return {"model": "deepseek-flash", "max_tokens": 512, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "summarize"}]}


def test_account_without_verified_prices_never_launches_cli(tmp_path, monkeypatch):
    monkeypatch.setattr("app.evaluation.deepseek_cli.subprocess.run", lambda *a, **k: pytest.fail("CLI must not launch"))
    ledger = BudgetLedger(tmp_path / "ledger.db")
    with pytest.raises(BudgetError) as caught:
        run_task("summarize", task_id="no-prices", ledger=ledger)
    assert caught.value.code == "account_pricing_unverified"
    assert not ledger.summary()["attempts"]


def test_recording_only_account_mode_needs_no_price_and_keeps_token_usage(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.db")
    ledger.set_enforcement(False)
    assert load_price(None, "account", required=ledger.is_enforced()) is None
    bridge = BudgetBridge(ledger, None, mode="account")
    identifier = bridge.handle("reserve", {"payload": payload()})["id"]
    body = json.dumps({"usage": {"input_tokens": 10, "output_tokens": 2}, "content": [{"type": "text", "text": "ok"}]})
    assert bridge.handle("complete", {"id": identifier, "status": 200, "body": body, "content_type": "application/json"})["accepted"]
    summary = ledger.summary()
    assert summary["attempts"][0]["input_tokens"] == 10
    assert summary["total_cost_cny"] is None


def test_prices_must_be_explicit_current_account_cny_records(tmp_path):
    path = tmp_path / "pricing.json"
    today = datetime.now(timezone(timedelta(hours=8))).date()
    record = {"verified": True, "auth_mode": "account", "currency": "CNY",
              "source_url": "https://platform.deepseek.com/test-fixture", "verified_on": str(today), "price": PRICE.__dict__}
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_price(path, "account") == PRICE
    for change in [{"verified": False}, {"currency": "USD"}, {"auth_mode": "api_key"},
                   {"source_url": "https://other.example"}, {"verified_on": str(today - timedelta(days=8))}]:
        path.write_text(json.dumps(record | change), encoding="utf-8")
        with pytest.raises(BudgetError):
            load_price(path, "account")


def test_cli_request_caps_share_global_ledger_and_reject_before_second_call(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.db")
    bridge = BudgetBridge(ledger, PRICE, mode="api_key")
    first = bridge.handle("reserve", {"payload": payload()})
    with pytest.raises(BudgetError):
        bridge.handle("reserve", {"payload": payload()})
    assert len(ledger.summary()["attempts"]) == 1
    body = json.dumps({"usage": {"input_tokens": 10, "cache_read_input_tokens": 1, "output_tokens": 2}, "content": [{"type": "text", "text": "ok"}]})
    assert bridge.handle("complete", {"id": first["id"], "status": 200, "body": body, "content_type": "application/json"})["accepted"]
    assert ledger.summary()["charged_cny"] == "0.00003604"


@pytest.mark.parametrize("change", [
    {"model": "unpriced"}, {"max_tokens": 9999}, {"thinking": {"type": "enabled"}},
    {"tools": []}, {"messages": [{"content": [{"type": "image", "source": {}}]}]},
])
def test_non_text_or_unbounded_cli_requests_refused(tmp_path, change):
    ledger = BudgetLedger(tmp_path / "ledger.db")
    with pytest.raises(BudgetError):
        BudgetBridge(ledger, PRICE, mode="api_key").handle("reserve", {"payload": payload() | change})
    assert not ledger.summary()["attempts"]


def test_partial_sse_does_not_turn_partial_usage_into_final_charge():
    start = 'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
    delta = 'data: {"type":"message_delta","usage":{"output_tokens":5}}\n\n'
    stop = 'data: {"type":"message_stop"}\n\n'
    assert response_usage(start + delta, "text/event-stream")[0] is None
    usage, tools = response_usage(start + delta + stop, "text/event-stream")
    assert usage["completion_tokens"] == 5 and usage["prompt_tokens"] == 10 and not tools
    assert response_usage('{"content":[{"type":"tool_use"}]}', "application/json")[1]


def test_cli_task_cost_limit_refuses_before_reserving_global_funds(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.db")
    bridge = BudgetBridge(ledger, PRICE, mode="account", task_cap_cny="0.0001")
    with pytest.raises(BudgetError) as caught:
        bridge.handle("reserve", {"payload": payload()})
    assert caught.value.code == "cli_task_budget" and not ledger.summary()["attempts"]


def test_tool_response_is_charged_but_never_accepted_for_execution(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.db")
    bridge = BudgetBridge(ledger, PRICE, mode="account")
    identifier = bridge.handle("reserve", {"payload": payload()})["id"]
    body = json.dumps({"usage": {"input_tokens": 10, "output_tokens": 2}, "content": [{"type": "tool_use", "name": "shell"}]})
    assert not bridge.handle("complete", {"id": identifier, "status": 200, "body": body, "content_type": "application/json"})["accepted"]
    assert ledger.summary()["attempts"][0]["state"] == "settled"


@pytest.fixture
def mock_node_cli(tmp_path, monkeypatch):
    if not shutil.which("node"):
        pytest.skip("Node.js required for the CLI supervisor")
    # Runs a real Node subprocess and real local budget HTTP server. ALL remote
    # fetches are mocked before the production guard loads; no provider traffic.
    preloader = tmp_path / "mock-provider.mjs"
    requests_path = tmp_path / "requests.jsonl"
    preloader.write_text('''
import {appendFileSync} from 'node:fs';
const localFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url;
  if (new URL(url).hostname === '127.0.0.1') return localFetch(input, init);
  if (new URL(url).origin !== 'https://api.deepseek.com') throw Error('unexpected destination');
  const body = JSON.parse(init.body);
  appendFileSync(process.env.RESEARCH_TEST_REQUESTS, JSON.stringify(body)+'\\n');
  if (body.tools || body.dsh_session_log || body.thinking.type !== 'disabled') throw Error('guard failed');
  if (process.env.RESEARCH_TEST_TIMEOUT === '1') throw Error('simulated transport failure');
  return new Response(JSON.stringify({usage:{input_tokens:10,output_tokens:5},content:[{type:'text',text:'mock answer'}]}),
                      {status:200,headers:{'content-type':'application/json'}});
};
''', encoding="utf-8")
    entry = tmp_path / "fake-dsh.mjs"
    entry.write_text('''
import {readFileSync} from 'node:fs';
const prompt = readFileSync(0,'utf8');
for(let i=0;i<Number(process.env.RESEARCH_TEST_CALLS || '1');i++) {
  const response = await fetch('https://api.deepseek.com/anthropic/v1/messages', {
    method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({model:'deepseek-flash',max_tokens:512,thinking:{type:'enabled'},
      messages:[{role:'user',content:prompt}],tools:[{name:'shell'}],dsh_session_log:'not transmitted'})
  });
  await response.json();
}
console.log('[llm-deepseek-account] model listing from https://api.deepseek.com/models loaded 1 model(s), skipped 0 row(s)');
console.log(JSON.stringify({type:'final',text:'mock answer'}));
''', encoding="utf-8")
    monkeypatch.setenv("NODE_OPTIONS", "--import " + preloader.as_uri())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unit-test")
    monkeypatch.setenv("RESEARCH_TEST_REQUESTS", str(requests_path))
    return entry, requests_path


def test_node_guard_accounts_and_resume_does_not_spawn_or_bill_twice(tmp_path, mock_node_cli):
    entry, requests_path = mock_node_cli
    ledger = BudgetLedger(tmp_path / "ledger.db")
    prompt = 'Literal input: $(Get-Content .env); `whoami`; do not execute.'
    answer = run_task(prompt, task_id="task-1", ledger=ledger, mode="api_key", entry=entry)
    assert answer["text"] == "mock answer" and not answer["resumed"]
    assert len(answer["attempt_ids"]) == 1
    repeated = run_task(prompt, task_id="task-1", ledger=ledger, mode="api_key", entry=entry)
    assert repeated["resumed"]
    outbound = [json.loads(line) for line in requests_path.read_text("utf-8").splitlines()]
    assert len(outbound) == 1
    assert outbound[0]["messages"][0]["content"] == prompt
    assert ledger.summary()["attempts"][0]["state"] == "settled"
    with pytest.raises(BudgetError, match="different input"):
        run_task("changed prompt", task_id="task-1", ledger=ledger, mode="api_key", entry=entry)


def test_node_guard_stops_hidden_second_request_before_remote_fetch(tmp_path, mock_node_cli, monkeypatch):
    entry, requests_path = mock_node_cli
    monkeypatch.setenv("RESEARCH_TEST_CALLS", "2")
    ledger = BudgetLedger(tmp_path / "ledger.db")
    with pytest.raises(BudgetError):
        run_task("summarize", task_id="bounded", ledger=ledger, mode="api_key", entry=entry)
    assert len(requests_path.read_text("utf-8").splitlines()) == 1
    assert len(ledger.summary()["attempts"]) == 1


def test_cli_transport_failure_preserves_reservation_and_refuses_automatic_rerun(tmp_path, mock_node_cli, monkeypatch):
    entry, _ = mock_node_cli
    monkeypatch.setenv("RESEARCH_TEST_TIMEOUT", "1")
    ledger = BudgetLedger(tmp_path / "ledger.db")
    for _ in range(2):
        with pytest.raises(BudgetError):
            run_task("summarize", task_id="failed", ledger=ledger, mode="api_key", entry=entry)
    assert len(ledger.summary()["attempts"]) == 1
    assert ledger.summary()["attempts"][0]["state"] == "unknown"
