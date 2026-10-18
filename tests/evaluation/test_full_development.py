import importlib.util
from pathlib import Path

import pytest


def verifier():
    spec = importlib.util.spec_from_file_location("verify_full_dev", Path("scripts/verify_full_dev.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_development_publication_recomputes_without_credentials():
    traces, valid, attempts = verifier().verify()
    assert traces == 80 and 0 < valid <= 80 and attempts >= 80


def test_unpriced_account_requests_cannot_be_reported_as_zero_cost():
    module = verifier()
    row = dict(input_tokens=20, output_tokens=10, cached_input_tokens=0,
               state="unpriced", price=None, estimated_cost_cny=None)
    aggregate = dict(physical_attempts=1, usage_complete=True, input_tokens=20, output_tokens=10,
                     cached_input_tokens=0, estimated_cost_cny=None, unknown_usage_attempts=0)
    module.check_usage(aggregate, [row])
    with pytest.raises(ValueError, match="usage aggregate differs"):
        module.check_usage(aggregate | {"estimated_cost_cny": "0"}, [row])
