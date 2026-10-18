"""Tracked text-only dsh tasks with optional monetary enforcement.

Run ``python -m app.evaluation.deepseek_cli --help`` for the local supervisor.
No credentials or response bodies are written into the accounting service logs.
"""

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock, Thread
from urllib.parse import urlparse

from app.config import Settings
from app.evaluation.budget import BudgetError, BudgetLedger, Price, PRICES, amount
from app.providers.accounting import RequestAccounting


def load_price(path: Path | None, mode: str, *, required: bool = True) -> Price | None:
    if mode == "api_key":
        return PRICES[("deepseek", "deepseek-flash")]
    if path is None:
        if not required:
            return None
        raise BudgetError("account_pricing_unverified", "confirm account-route CNY pricing before paid CLI tasks")
    record = json.loads(path.read_text("utf-8"))
    if record.get("verified") is not True or record.get("auth_mode") != "account" or record.get("currency") != "CNY":
        raise BudgetError("account_pricing_unverified", "account pricing record is not verified in CNY")
    source = urlparse(record.get("source_url", ""))
    today = datetime.now(timezone(timedelta(hours=8))).date()
    age = (today - date.fromisoformat(record["verified_on"])).days
    if source.scheme != "https" or source.hostname not in {"platform.deepseek.com", "api-docs.deepseek.com"} or not 0 <= age <= 7:
        raise BudgetError("account_pricing_unverified", "pricing source or verification date is invalid")
    return Price(**record["price"])


def response_usage(body: str, content_type: str) -> tuple[dict | None, bool]:
    """Messages usage is cumulative. Cache input is separate from uncached input.

    Incomplete streams retain the full reservation even if an early usage event
    was present. Tool calls are rejected before the CLI receives any response.
    """
    events = []
    complete = "text/event-stream" not in content_type
    try:
        if not complete:
            for line in body.splitlines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
            complete = any(event.get("type") == "message_stop" for event in events)
        else:
            events = [json.loads(body)]
        usage = {}
        tool_call = False
        for event in events:
            message = event.get("message", {})
            for part in [event.get("content_block", {})] + event.get("content", []) + message.get("content", []):
                if isinstance(part, dict) and part.get("type") in {"tool_use", "server_tool_use"}:
                    tool_call = True
            if event.get("type") == "error":
                complete = False
            usage.update(message.get("usage", {}))
            usage.update(event.get("usage", {}))
        if not complete or not {"input_tokens", "output_tokens"} <= usage.keys():
            return None, tool_call
        values = [usage.get(key, 0) for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")]
        if any(type(value) is not int or value < 0 for value in values):
            return None, tool_call
        input_tokens, output_tokens, read, created = values
        return {"prompt_tokens": input_tokens + read + created, "completion_tokens": output_tokens,
                "prompt_cache_hit_tokens": read}, tool_call
    except (ValueError, TypeError, AttributeError):
        return None, True


class BudgetBridge:
    def __init__(self, ledger: BudgetLedger, price: Price | None, *, mode: str,
                 max_requests: int = 1, max_output: int = 512, task_cap_cny: str = "0.5"):
        if not 1 <= max_requests <= 3 or not 1 <= max_output <= 2048:
            raise ValueError("CLI limits out of range")
        self.ledger, self.price = ledger, price
        self.accounting = RequestAccounting(ledger, "deepseek-account" if mode == "account" else "deepseek",
                                            "deepseek-flash", "cli", "preparation", price)
        self.max_requests, self.max_output = max_requests, max_output
        self.task_cap = amount(task_cap_cny)
        if not 0 < self.task_cap <= amount("5"):
            raise ValueError("CLI task cap must be positive and at most 5 CNY")
        self.lock, self.calls, self.catalog_calls = Lock(), 0, 0

    def handle(self, operation: str, data: dict) -> dict:
        with self.lock:
            if operation == "catalog":
                if self.catalog_calls >= 1:
                    raise BudgetError("cli_catalog_limit", "model catalog limit reached")
                self.catalog_calls += 1
                return {"accepted": True}
            if operation == "reserve":
                payload = data["payload"]
                if payload.get("model") != "deepseek-flash" or type(payload.get("max_tokens")) is not int or not 1 <= payload["max_tokens"] <= self.max_output:
                    raise BudgetError("cli_request_limit", "model or output limit does not match task")
                if payload.get("thinking") != {"type": "disabled"} or "tools" in payload:
                    raise BudgetError("cli_request_limit", "helper must use non-thinking text-only mode")
                # Reject images, tools and remote files even inside message blocks.
                def text_only(node):
                    if isinstance(node, dict):
                        if node.get("type") in {"image", "document", "file", "tool_use", "tool_result", "server_tool_use"}:
                            return False
                        return all(text_only(value) for value in node.values())
                    return all(text_only(value) for value in node) if isinstance(node, list) else True
                if not text_only(payload):
                    raise BudgetError("cli_request_limit", "non-text content is not allowed")
                bound = len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")) + 4096
                rows = self.ledger.summary()["attempts"]
                exposure = sum(row["reserved"] if row["state"] in {"reserved", "unknown"} else row["charged"]
                               for row in rows if row["id"] in self.accounting.ids)
                if self.calls >= self.max_requests:
                    raise BudgetError("cli_request_limit", "helper exceeded its declared request count")
                if self.ledger.is_enforced() and self.price is not None and exposure + self.price.cost(bound, payload["max_tokens"]) > self.task_cap:
                    raise BudgetError("cli_task_budget", "task request or cost limit reached")
                identifier = self.accounting.reserve(payload, payload["max_tokens"])
                self.calls += 1
                return {"id": identifier}
            identifier = data.get("id")
            if identifier not in self.accounting.ids:
                raise BudgetError("cli_unknown_attempt", "attempt does not belong to this task")
            if operation == "unknown":
                self.accounting.unknown(identifier)
                return {"accepted": True}
            if operation == "complete":
                usage, tool_call = response_usage(data["body"], data["content_type"])
                self.accounting.record(identifier, usage)
                accepted = 200 <= data["status"] < 300 and (usage is not None or not self.ledger.is_enforced()) and not tool_call
                outcome = "success" if accepted else "cli_response_rejected"
                self.accounting.annotate(identifier, outcome)
                return {"accepted": accepted, "code": outcome}
            raise BudgetError("cli_unknown_operation", "unknown operation")


@contextmanager
def serve_bridge(bridge: BudgetBridge):
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            status = 200
            try:
                if self.headers.get("Authorization") != f"Bearer {token}":
                    raise BudgetError("cli_bridge_auth", "local authorization failed")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 3 * 1024 * 1024:
                    raise ValueError("invalid body length")
                data = json.loads(self.rfile.read(length))
                result = bridge.handle(self.path.lstrip("/"), data)
            except BudgetError as error:
                status, result = 409, {"code": error.code}
            except Exception:
                status, result = 400, {"code": "cli_bridge_invalid_request"}
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", token
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _entry_path() -> Path:
    command = shutil.which("dsh.cmd") or shutil.which("dsh")
    if command is None:
        raise ValueError("official dsh CLI not found")
    path = Path(command)
    if path.suffix.lower() in {".cmd", ".ps1"}:
        path = path.parent / "node_modules/@deepseek-ai/dsh/lib/bin.js"
    return path.resolve(strict=True)


def _patch(mode: str, max_output: int) -> str:
    # Private per-task overlay; the user's saved profile is not changed.
    rows = [
        {"id": "agent-default-model", "config": {"provider": "deepseek-account" if mode == "account" else "deepseek-official", "model": "deepseek-flash", "reasoningEffort": "off"}},
        {"id": "llm-deepseek-account" if mode == "account" else "llm-deepseek", "config": {
            "baseURL": "https://api.deepseek.com/anthropic", "thinking": "disabled", "reasoningEffort": "off",
            "maxTokens": max_output, "retryPolicy": {"mode": "normal", "maxRetries": 0},
        }},
    ]
    for identifier in ["credentials", "authorization", "deepseek-account"]:
        rows.append({"id": identifier, "disabled": mode != "account"})
    for identifier in ["session-title-llm", "session-log-deepseek", "plugin-package-inventory-deepseek",
                       "tool-bash", "tool-pwsh", "tool-jobs", "tool-fs", "tool-fs-search", "tool-skill",
                       "tool-subagent-control", "tool-subagent-list-agents", "tool-subagent", "tool-subagent-fork",
                       "tool-workflow", "tool-todo", "tool-goal", "tool-web", "tool-plugin-manager"]:
        rows.append({"id": identifier, "disabled": True})
    # JSON is a YAML subset and contains no credential values.
    return json.dumps(rows, indent=2)


def run_task(prompt: str, *, task_id: str, ledger: BudgetLedger, mode: str = "account",
             price_file: Path | None = None, max_requests: int = 1, max_output: int = 512,
             task_cap_cny: str = "0.5", timeout: float = 90, entry: Path | None = None,
             max_prompt_bytes: int = 32000) -> dict:
    price = load_price(price_file, mode, required=ledger.is_enforced())
    if type(max_prompt_bytes) is not int or not 1 <= max_prompt_bytes <= 131072:
        raise ValueError("invalid task prompt limit")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", task_id) or not 0 < len(prompt.encode("utf-8")) <= max_prompt_bytes:
        raise ValueError("invalid task id or prompt size")
    if mode not in {"account", "api_key"} or not 1 <= timeout <= 180:
        raise ValueError("invalid CLI mode or deadline")
    node, cli_entry = shutil.which("node"), entry or _entry_path()
    if not node:
        raise ValueError("Node.js is required")
    environment = os.environ.copy()
    if mode == "api_key":
        configured = Settings(_env_file=Path.cwd() / ".env").deepseek_api_key
        if configured is None or not configured.get_secret_value().strip():
            raise BudgetError("cli_missing_api_key", "configure the CLI API key locally before launch")
        environment["DEEPSEEK_API_KEY"] = configured.get_secret_value()
    bridge = BudgetBridge(ledger, price, mode=mode, max_requests=max_requests,
                          max_output=max_output, task_cap_cny=task_cap_cny)
    identity = {"prompt": prompt, "mode": mode, "price": price.__dict__ if price is not None else None,
                "monetary_cap_enforced": ledger.is_enforced(), "max_requests": max_requests,
                "max_output": max_output, "max_prompt_bytes": max_prompt_bytes, "task_cap_cny": task_cap_cny,
                "cli_sha256": sha256(cli_entry.read_bytes()).hexdigest(),
                "guard_sha256": sha256(Path(__file__).with_name("dsh_guard.mjs").read_bytes()).hexdigest(),
                "supervisor_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    digest = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Share the durable SQLite transaction domain with the request ledger.
    with ledger._transaction() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS cli_tasks (id TEXT PRIMARY KEY, digest TEXT, state TEXT, result TEXT)")
        previous = connection.execute("SELECT * FROM cli_tasks WHERE id=?", (task_id,)).fetchone()
        if previous is not None:
            if previous["digest"] != digest:
                raise BudgetError("cli_task_mismatch", "task id already names a different input/configuration")
            if previous["state"] == "complete":
                return json.loads(previous["result"]) | {"resumed": True}
            raise BudgetError("cli_task_incomplete", "prior task is incomplete; reconcile it before a new attempt")
        connection.execute("INSERT INTO cli_tasks VALUES (?,?,'running',NULL)", (task_id, digest))
    try:
        with TemporaryDirectory(prefix="research-dsh-") as temporary, serve_bridge(bridge) as (url, token):
            workspace = Path(temporary)
            patch = workspace / "budget.patch.yml"
            patch.write_text(_patch(mode, max_output), encoding="utf-8")
            environment.update(RESEARCH_BUDGET_BRIDGE=url, RESEARCH_BUDGET_TOKEN=token)
            # API keys are not copied from the credential store or put in argv.
            result = subprocess.run([
                node, "--import", Path(__file__).with_name("dsh_guard.mjs").as_uri(), str(cli_entry),
                "--profile", "headless", "--patch", str(patch), "--json", "-",
            ], input=prompt, text=True, encoding="utf-8", capture_output=True,
                cwd=workspace, env=environment, timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode != 0 or len(result.stdout.encode("utf-8")) > 2 * 1024 * 1024:
                raise BudgetError("cli_task_failed", "CLI task failed; inspect ledger status, not raw credential-bearing diagnostics")
            # dsh 0.2.1-rc.2 writes an account-catalog diagnostic to stdout
            # even in --json mode. Ignore that known prefix, not malformed events.
            events = [json.loads(line) for line in result.stdout.splitlines()
                      if line.strip() and not line.startswith("[llm-deepseek-account] ")]
            final = [event for event in events if event.get("type") == "final" and isinstance(event.get("text"), str)]
            if not final or not bridge.calls:
                raise BudgetError("cli_invalid_output", "CLI returned no accounted final answer")
            answer = {"task_id": task_id, "input_sha256": digest, "text": final[-1]["text"],
                      "attempt_ids": bridge.accounting.ids, "resumed": False}
        with ledger._transaction() as connection:
            connection.execute("UPDATE cli_tasks SET state='complete',result=? WHERE id=?", (json.dumps(answer), task_id))
        return answer
    except BaseException:
        for identifier in bridge.accounting.ids:
            ledger.unknown(identifier, "cli_interrupted")
        with ledger._transaction() as connection:
            connection.execute("UPDATE cli_tasks SET state='failed' WHERE id=?", (task_id,))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one tracked text-only DeepSeek CLI helper task.")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--mode", choices=("account", "api_key"), default="account")
    parser.add_argument("--account-price-file", type=Path)
    parser.add_argument("--ledger", type=Path, default=Path("artifacts/evaluation/budget/ledger.sqlite"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--task-cap-cny", default="0.5")
    parser.add_argument("--enforce-budget", action="store_true", help="optional monetary cap; requires verified account pricing")
    args = parser.parse_args(argv)
    try:
        ledger = BudgetLedger(args.ledger)
        ledger.set_enforcement(args.enforce_budget)
        answer = run_task(args.prompt_file.read_text("utf-8"), task_id=args.task_id, ledger=ledger,
                          mode=args.mode, price_file=args.account_price_file, max_requests=args.max_requests,
                          max_output=args.max_output_tokens, task_cap_cny=args.task_cap_cny)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"CLI task saved; {len(answer['attempt_ids'])} accounted request(s); resumed={answer['resumed']}.")
        return 0
    except BudgetError as error:
        print(f"CLI stopped ({error.code}).")
        return 5
    except Exception as error:
        print(f"CLI execution error ({type(error).__name__}).")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
