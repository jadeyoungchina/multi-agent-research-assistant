from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from hashlib import sha256
import json

import pytest

from app.evaluation.budget import BudgetError, BudgetLedger, Price, amount


PRICE = Price("2", "8", "0.04", "test-v1", 1_000_000)


def reserve(ledger, **changes):
    arguments = dict(provider="deepseek", model="deepseek-flash", kind="chat", role="preparation",
                     request_hash=sha256(b"private prompt is never stored").hexdigest(), price=PRICE,
                     max_input=1000, max_output=1000)
    return ledger.reserve(**(arguments | changes))


def test_decimal_cost_and_cached_input_are_not_double_counted():
    assert PRICE.cost(1000, 1000) == amount("0.01")
    assert PRICE.cost(1000, 1000, 500) == amount("0.00902")
    assert amount(Decimal("0.0000000001")) == 1
    for invalid in ("NaN", "Infinity", "-1", 0.1):
        with pytest.raises((ValueError, TypeError)):
            amount(invalid)
    with pytest.raises(ValueError):
        PRICE.cost(1, 1, 2)


def test_reservation_settlement_and_unknown_survive_reopen(tmp_path):
    path = tmp_path / "budget.db"
    ledger = BudgetLedger(path)
    first, second = reserve(ledger), reserve(ledger)
    ledger.settle(first, input_tokens=100, output_tokens=200)
    ledger.unknown(second, "timeout")
    summary = BudgetLedger(path).summary()
    assert summary["charged_cny"] == "0.0018"
    assert summary["reserved_or_unknown_cny"] == "0.01"
    assert [a["state"] for a in summary["attempts"]] == ["settled", "unknown"]
    assert "private prompt" not in json.dumps(summary)
    with pytest.raises(BudgetError):
        ledger.settle(first, input_tokens=100, output_tokens=200)


def test_unknown_can_only_release_after_explicit_usage_reconciliation(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.db")
    identifier = reserve(ledger)
    ledger.unknown(identifier)
    ledger.settle(identifier, input_tokens=0, output_tokens=0, outcome="invoice_reconciled")
    assert ledger.summary()["exposure_cny"] == "0"


def test_separate_connections_atomically_reserve_last_available_budget(tmp_path):
    path = tmp_path / "budget.db"
    BudgetLedger(path, total_cny="0.01", stop_cny="0.01")
    def admit(_):
        try:
            reserve(BudgetLedger(path, total_cny="0.01", stop_cny="0.01"))
            return True
        except BudgetError as error:
            assert error.code == "budget_exhausted"
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(admit, range(8))) == 1
    assert len(BudgetLedger(path, total_cny="0.01", stop_cny="0.01").summary()["attempts"]) == 1


def test_policy_cannot_be_reset_by_reopening_or_invalid_limits(tmp_path):
    path = tmp_path / "budget.db"
    BudgetLedger(path)
    with pytest.raises(BudgetError, match="original budget"):
        BudgetLedger(path, total_cny="100", stop_cny="100")
    for total, stop in [("101", "90"), ("10", "11"), ("0", "0")]:
        with pytest.raises(ValueError):
            BudgetLedger(tmp_path / f"invalid-{total}.db", total_cny=total, stop_cny=stop)


def test_tier_invalid_labels_and_usage_rejected_before_charge(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.db")
    with pytest.raises(BudgetError):
        reserve(ledger, max_input=1_000_001)
    with pytest.raises(ValueError):
        reserve(ledger, role="sk-private-value")
    with pytest.raises(ValueError):
        reserve(ledger, max_output=-1)
    assert not ledger.summary()["attempts"]


def test_actual_overrun_stays_visible_and_blocks_admissions(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.db")
    identifier = reserve(ledger, max_input=1, max_output=1)
    ledger.settle(identifier, input_tokens=2000, output_tokens=2000)
    assert ledger.summary()["overrun_detected"]
    with pytest.raises(BudgetError):
        reserve(ledger)
