"""Cross-process request reservations. Amounts are integer nano-CNY, never floats.

The ledger contains labels, hashes and usage, never prompts or credentials.
Reservations survive process death; unknown charges are never silently released.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from app.domain.errors import DomainError


SCALE = 1_000_000_000


class BudgetError(DomainError):
    pass


def amount(value: str | Decimal) -> int:
    if not isinstance(value, (str, Decimal)):
        raise ValueError("money must be a decimal string")
    number = Decimal(value)
    if not number.is_finite() or number < 0:
        raise ValueError("money must be finite and nonnegative")
    return int((number * SCALE).to_integral_value(rounding=ROUND_CEILING))


def cny(nano: int) -> str:
    return format(Decimal(nano) / SCALE, "f")


def _count(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("usage must contain nonnegative integers")
    return value


def _label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", value) or "sk-" in value:
        raise ValueError("invalid ledger label")
    return value


@dataclass(frozen=True)
class Price:
    """A dated upper-bound price in CNY per million tokens."""

    input: str
    output: str
    cached_input: str
    version: str
    max_input_tokens: int

    def __post_init__(self):
        for value in (self.input, self.output, self.cached_input):
            amount(value)
        if Decimal(self.cached_input) > Decimal(self.input):
            raise ValueError("cache price exceeds input price")
        _label(self.version)
        if _count(self.max_input_tokens) == 0:
            raise ValueError("input token limit must be positive")

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> int:
        for count in (input_tokens, output_tokens, cached_tokens):
            _count(count)
        if cached_tokens > input_tokens:
            raise ValueError("cached tokens exceed total input")
        value = (
            Decimal(self.input) * (input_tokens - cached_tokens)
            + Decimal(self.cached_input) * cached_tokens
            + Decimal(self.output) * output_tokens
        ) / 1_000_000
        return amount(value)


# Conservative peak/no-discount prices. Exceeding a tier is rejected before I/O.
# CLI account authentication is intentionally NOT assigned an API-key price.
PRICES = {
    ("dashscope", "qwen3.7-flash"): Price("0.2", "0.8", "0.2", "2026-10-17-beijing-le32k", 32000),
    ("dashscope", "qwen3.7-flash-2026-07-15"): Price("0.2", "0.8", "0.2", "2026-10-17-beijing-le32k", 32000),
    ("dashscope", "text-embedding-v4"): Price("0.5", "0", "0.5", "2026-10-17-beijing", 1_000_000),
    ("deepseek", "deepseek-flash"): Price("2", "8", "0.04", "2026-10-17-api-key-peak", 1_000_000),
}


class BudgetLedger:
    def __init__(self, path: Path, *, total_cny: str = "100", stop_cny: str = "90"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        total, stop = amount(total_cny), amount(stop_cny)
        if not 0 < stop <= total <= amount("100"):
            raise ValueError("require 0 < stop <= total <= 100 CNY")
        with self._transaction() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY CHECK(id=1), total INTEGER, stop INTEGER)")
            connection.execute("""CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                provider TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL, role TEXT NOT NULL,
                request_hash TEXT NOT NULL, price_json TEXT NOT NULL,
                reserved INTEGER NOT NULL, charged INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL, outcome TEXT NOT NULL,
                input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER)""")
            connection.execute("INSERT OR IGNORE INTO policy VALUES (1, ?, ?)", (total, stop))
            stored = connection.execute("SELECT total, stop FROM policy WHERE id=1").fetchone()
            if tuple(stored) != (total, stop):
                raise BudgetError("budget_policy_mismatch", "reuse the ledger's original budget policy")

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _exposure(connection) -> tuple[int, int]:
        row = connection.execute("""SELECT COALESCE(SUM(charged),0),
            COALESCE(SUM(CASE WHEN state IN ('reserved','unknown') THEN reserved ELSE 0 END),0)
            FROM attempts""").fetchone()
        return row[0], row[1]

    def reserve(self, *, provider: str, model: str, kind: str, role: str,
                request_hash: str, price: Price, max_input: int, max_output: int) -> str:
        for label in (provider, model, kind, role):
            _label(label)
        if not re.fullmatch(r"[0-9a-f]{64}", request_hash):
            raise ValueError("request hash must be SHA-256")
        if _count(max_input) > price.max_input_tokens:
            raise BudgetError("budget_input_limit", "request exceeds the priced input tier")
        reserved = price.cost(max_input, _count(max_output))
        identifier = uuid4().hex
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM attempts WHERE state='overrun' LIMIT 1").fetchone():
                raise BudgetError("budget_overrun", "provider usage exceeded its reservation; reconcile before continuing")
            charged, pending = self._exposure(connection)
            stop = connection.execute("SELECT stop FROM policy WHERE id=1").fetchone()[0]
            if charged + pending >= stop or charged + pending + reserved > stop:
                raise BudgetError("budget_exhausted", "budget cannot admit another request")
            connection.execute("""INSERT INTO attempts
                (id,created_at,provider,model,kind,role,request_hash,price_json,reserved,state,outcome)
                VALUES (?,?,?,?,?,?,?,?,?,'reserved','in_flight')""", (
                    identifier, datetime.now(timezone.utc).isoformat(), provider, model, kind, role,
                    request_hash, json.dumps(price.__dict__, sort_keys=True), reserved,
                ))
        return identifier

    def settle(self, identifier: str, *, input_tokens: int, output_tokens: int,
               cached_tokens: int = 0, outcome: str = "response") -> None:
        _label(outcome)
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM attempts WHERE id=?", (identifier,)).fetchone()
            if row is None or row["state"] not in ("reserved", "unknown"):
                raise BudgetError("budget_attempt_closed", "attempt is absent or already settled")
            cost = Price(**json.loads(row["price_json"])).cost(input_tokens, output_tokens, cached_tokens)
            # Actual provider usage is authoritative even when it exceeds our bound.
            # Keep the overrun visible; all subsequent admissions stop.
            state = "overrun" if cost > row["reserved"] else "settled"
            connection.execute("""UPDATE attempts SET state=?,charged=?,input_tokens=?,
                output_tokens=?,cached_tokens=?,outcome=? WHERE id=?""",
                (state, cost, input_tokens, output_tokens, cached_tokens, outcome, identifier))

    def unknown(self, identifier: str, outcome: str = "usage_unknown") -> None:
        _label(outcome)
        with self._transaction() as connection:
            connection.execute("UPDATE attempts SET state='unknown',outcome=? WHERE id=? AND state='reserved'",
                               (outcome, identifier))

    def annotate(self, identifier: str, outcome: str) -> None:
        with self._transaction() as connection:
            connection.execute("UPDATE attempts SET outcome=? WHERE id=?", (_label(outcome), identifier))

    def summary(self) -> dict:
        with self._transaction() as connection:
            charged, pending = self._exposure(connection)
            policy = connection.execute("SELECT * FROM policy WHERE id=1").fetchone()
            rows = connection.execute("SELECT * FROM attempts ORDER BY created_at,id").fetchall()
            overrun = any(row["state"] == "overrun" for row in rows)
            return {
                "currency": "CNY", "total_cny": cny(policy["total"]), "stop_cny": cny(policy["stop"]),
                "charged_cny": cny(charged), "reserved_or_unknown_cny": cny(pending),
                "exposure_cny": cny(charged + pending), "reestimate_required": charged + pending >= amount("70"),
                "overrun_detected": overrun, "attempts": [dict(row) for row in rows],
            }
