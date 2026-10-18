import importlib.util
import json
from pathlib import Path

import pytest

from app.evaluation.real_dataset import validate_real_benchmark


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit_real_references", ROOT / "scripts/audit_real_references.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_audit_uses_complete_sources_and_covers_every_negative_in_its_split():
    root = ROOT / "benchmarks/real-v1"
    tasks = audit.jobs(root)
    families = [task for task in tasks if task[1] == "references"]
    assert len(families) == 12
    assert sum(len(task[2]) for task in families) == 48
    dev, test = validate_real_benchmark(root)
    for suite in (dev, test):
        split = suite.cases[0].split
        negatives = {case.id for case in suite.cases if not case.expected_sufficiency}
        shards = [task for task in tasks if task[0].startswith(f"absence-{split}-")]
        for _, _, cases, prompt in shards:
            assert {case.id for case in cases} == negatives
            assert len(prompt.encode()) <= 131072
        for doc in suite.manifest.documents:
            if doc.split == split:
                text = (root / doc.corpus_path).read_text("utf-8")
                assert sum(text in task[3] for task in shards) == 1


def test_reference_hash_tracks_content_but_not_human_status():
    dev, _ = validate_real_benchmark(ROOT / "benchmarks/real-v1")
    case = dev.cases[0]
    changed = case.model_copy(update={"question": case.question + " changed"})
    assert audit.reference_digest(case) != audit.reference_digest(changed)
    changed = case.model_copy(update={"human_review": case.human_review.model_copy(update={"notes": "review note"})})
    assert audit.reference_digest(case) == audit.reference_digest(changed)


def test_fabricated_reviewer_quotes_are_rejected():
    dev, _ = validate_real_benchmark(ROOT / "benchmarks/real-v1")
    case = dev.cases[0]
    payload = [dict(id=case.id, verdict="revise", issues=["test finding"], recommendations=[],
                    suggested_evidence=[dict(source_file=case.expected_evidence[0].source_file, quote="a fabricated review quote")])]
    with pytest.raises(ValueError, match="verbatim"):
        audit.parse_response(json.dumps(payload), "references", [case], ROOT / "benchmarks/real-v1")


def test_duplicate_or_missing_reviews_are_rejected():
    dev, _ = validate_real_benchmark(ROOT / "benchmarks/real-v1")
    case = dev.cases[0]
    payload = [dict(id=case.id, verdict="pass", issues=[], recommendations=[], suggested_evidence=[])] * 2
    with pytest.raises(ValueError, match="exactly once"):
        audit.parse_response(json.dumps(payload), "references", [case], ROOT / "benchmarks/real-v1")


@pytest.mark.parametrize("value", [0, -1, 131073, True])
def test_helper_rejects_invalid_explicit_prompt_limits(value, tmp_path):
    from app.evaluation.budget import BudgetLedger
    from app.evaluation.deepseek_cli import run_task
    ledger = BudgetLedger(tmp_path / "audit.sqlite")
    ledger.set_enforcement(False)
    with pytest.raises(ValueError, match="prompt limit"):
        run_task("text", task_id="invalid-limit", ledger=ledger, max_prompt_bytes=value)
