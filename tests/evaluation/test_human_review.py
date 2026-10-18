import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("review_real_dataset", ROOT / "scripts/review_real_dataset.py")
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)


@pytest.fixture
def review_fixture(tmp_path):
    root = tmp_path / "real-v1"
    shutil.copytree(ROOT / "benchmarks/real-v1", root)
    payload = reviewer.review_payload(root)
    # Test-only synthetic reviewer; never applied to the actual benchmark files.
    export = dict(schema_version=payload["schema_version"], hashes=payload["hashes"], reviews=[
        dict(id=case["id"], **case["human_review"]) for case in payload["cases"]
    ])
    export["reviews"][0].update(status="approved", reviewer="Unit-test synthetic reviewer",
                                 reviewed_at="2026-10-18T12:00:00+08:00")
    path = tmp_path / "explicit-test-review.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    return root, path, export


def test_review_page_includes_original_context_and_distractors_without_approving():
    payload = reviewer.review_payload(ROOT / "benchmarks/real-v1")
    assert len(payload["cases"]) == 60
    assert all(case["human_review"]["status"] == "pending" for case in payload["cases"])
    assert len(payload["cases"][0]["candidate_pool"]) == 8
    assert "#L" in payload["cases"][0]["sources"][0]["url"]
    assert payload["cases"][0]["sources"][0]["quote"] in payload["cases"][0]["sources"][0]["context"]


def test_explicit_review_import_updates_only_the_selected_status(review_fixture):
    from app.evaluation.real_dataset import validate_real_benchmark
    root, path, _ = review_fixture
    assert reviewer.apply_review(root, path) == 1
    suites = validate_real_benchmark(root)
    assert sum(case.human_review.status == "approved" for suite in suites for case in suite.cases) == 1
    assert sum(case.human_review.status == "pending" for suite in suites for case in suite.cases) == 59


@pytest.mark.parametrize("kind", ["stale_hash", "blank_reviewer", "missing_timestamp", "duplicate", "missing_case"])
def test_invalid_review_is_rejected_before_either_split_is_changed(review_fixture, kind):
    root, path, export = review_fixture
    before = [(root / f"{split}.jsonl").read_bytes() for split in ("dev", "test")]
    if kind == "stale_hash":
        export["hashes"]["dev.jsonl"] = "0" * 64
    elif kind == "blank_reviewer":
        export["reviews"][0]["reviewer"] = " "
    elif kind == "missing_timestamp":
        export["reviews"][0]["reviewed_at"] = None
    elif kind == "duplicate":
        export["reviews"][1]["id"] = export["reviews"][0]["id"]
    else:
        export["reviews"].pop()
    path.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(ValueError):
        reviewer.apply_review(root, path)
    assert [(root / f"{split}.jsonl").read_bytes() for split in ("dev", "test")] == before
