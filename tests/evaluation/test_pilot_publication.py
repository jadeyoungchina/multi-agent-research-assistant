import importlib.util
from pathlib import Path


def test_public_usage_distinguishes_unknown_account_price_from_zero_cost():
    spec = importlib.util.spec_from_file_location("publish_dev_pilot", Path("scripts/publish_dev_pilot.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = dict(input_tokens=100, output_tokens=10, cached_tokens=20,
               price_json="null", state="unpriced", charged=0)
    result = module.usage([row])
    assert result["usage_complete"] and result["input_tokens"] == 100
    assert result["estimated_cost_cny"] is None
    priced = module.usage([row | dict(price_json="{}", state="settled", charged=1000000)])
    assert priced["estimated_cost_cny"] == "0.001"
    unknown = module.usage([row | dict(input_tokens=None, output_tokens=None, state="unknown")])
    assert not unknown["usage_complete"] and unknown["estimated_cost_cny"] is None
    assert unknown["unknown_usage_attempts"] == 1
