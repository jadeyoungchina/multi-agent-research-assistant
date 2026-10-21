import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest


def verifier():
    spec = importlib.util.spec_from_file_location("verify_full_dev", Path("scripts/verify_full_dev.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_development_publication_recomputes_without_credentials():
    traces, valid, attempts = verifier().verify()
    assert traces == 80 and 0 < valid <= 80 and attempts >= 80


def test_final_development_publication_recomputes_without_credentials():
    traces, valid, attempts = verifier().verify(Path("docs/evaluation/final-dev"))
    assert traces == valid == 80 and attempts >= 80


def test_source_review_count_cannot_be_inflated_in_summary(tmp_path):
    root = tmp_path / "packet"
    shutil.copytree("docs/evaluation/final-dev", root)
    path = root / "summary.json"
    summary = json.loads(path.read_text("utf-8"))
    summary["codex_source_checks"] += 1
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="source review attribution"):
        verifier().verify(root)


def test_corrected_judgment_cannot_be_relabelled_as_original_acceptance(tmp_path):
    root = tmp_path / "packet"
    shutil.copytree("docs/evaluation/final-dev", root)
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    audit = json.loads((root / "adjudications.json").read_text("utf-8"))
    judgments = json.loads((root / "ai-judgments.json").read_text("utf-8"))
    unit, decision = next((u, d) for u, d in audit["adjudications"].items() if d["action"] == "corrected")
    decision["action"] = "accepted"
    next(j for j in judgments if j["unit"] == unit)["codex_adjudication"] = decision
    summary["corrected_judgments"] -= 1
    for name, data in (("adjudications.json", audit), ("ai-judgments.json", judgments)):
        (root / name).write_text(json.dumps(data), encoding="utf-8")
    for item in summary["publications"]:
        item["sha256"] = sha256((root / item["path"]).read_bytes()).hexdigest()
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted review changes"):
        verifier().verify(root)


def test_unpriced_account_requests_cannot_be_reported_as_zero_cost():
    module = verifier()
    row = dict(input_tokens=20, output_tokens=10, cached_input_tokens=0,
               state="unpriced", price=None, estimated_cost_cny=None)
    aggregate = dict(physical_attempts=1, usage_complete=True, input_tokens=20, output_tokens=10,
                     cached_input_tokens=0, estimated_cost_cny=None, unknown_usage_attempts=0)
    module.check_usage(aggregate, [row])
    with pytest.raises(ValueError, match="usage aggregate differs"):
        module.check_usage(aggregate | {"estimated_cost_cny": "0"}, [row])
