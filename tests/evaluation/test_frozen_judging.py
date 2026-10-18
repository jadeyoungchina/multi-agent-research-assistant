from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from app.evaluation.development import block_judging_prompt, score_development
from app.evaluation.models import WorkflowVariant
from app.evaluation.real_dataset import load_real_suite
from app.evaluation.test_analysis import primary_comparison
from app.evaluation.test_judging import TestJudgment, score_test, test_prompt as make_prompt
from app.evaluation.trace import EvaluationTrace


def test_test_prompt_blinds_generator_metadata_and_uses_original_test_reference(monkeypatch):
    from app.evaluation import development
    monkeypatch.setattr(development, "load_scoring_cases", lambda *a: pytest.fail("test must not use dev overlay"))
    root = Path("benchmarks/real-v1")
    case = load_real_suite(root / "corpus-manifest.json", root / "test.jsonl", "test").cases[0]
    trace = EvaluationTrace(case_id=case.id, variant=WorkflowVariant.MULTI_AGENT_RAG, status="success",
        answer="A domain fact.", latency_ms=7703, prompt_tokens=12345, completion_tokens=300,
        model_calls=7, critic_loops=2, provider="private-generator-123", model="private-model-123")
    text = make_prompt(case, trace)
    data = json.loads(text.split("\nData:\n", 1)[1])
    assert set(data) == {"question", "reference_key_points", "expected_sufficiency", "absence_audit",
                         "reference_context", "answer_blocks", "answer_evidence"}
    assert data["reference_key_points"] == case.answer_key_points
    assert "private-generator-123" not in text and "private-model-123" not in text
    assert '"test-block-v2"' in text
    judgment = TestJudgment(rubric_version="test-block-v2",
        key_points=[{"index": i, "score": 2, "reason": "correct"} for i in range(len(case.answer_key_points))],
        blocks=[{"index": 0, "requires_citation": True, "support": "uncited", "reason": "no citation"}],
        abstained=False, unsupported_assertion=False, explanation="test-only grading")
    assert score_test(case, trace, judgment).answer_correctness == 1
    with pytest.raises(ValueError, match="development trace"):
        score_development(case, trace, judgment)


def test_primary_averages_repeats_then_pairs_questions_and_preserves_unknowns():
    cases = [SimpleNamespace(id="a", family="f1"), SimpleNamespace(id="b", family="f2")]
    values = {}
    for repeat in range(3):
        values["a", "multi_agent_rag", repeat] = 0.0  # Executed failures receive zero.
        values["a", "single_agent_rag", repeat] = 1.0
        values["b", "multi_agent_rag", repeat] = 1.0
        values["b", "single_agent_rag", repeat] = 0.0
    complete = primary_comparison(cases, values, replicates=100, seed=3)
    assert complete["complete_pairs"] == 2
    assert complete["mean_paired_correctness_difference"] == 0
    assert complete["all_question_missing_value_bounds"] == [0, 0]
    assert complete["family_count"] == 2  # Never six independent repeats.
    values["b", "multi_agent_rag", 0] = None  # Missing judge is not a failed answer.
    missing = primary_comparison(cases, values, replicates=100, seed=3)
    assert missing["complete_pairs"] == 1 and missing["missing_pairs"] == 1
    assert missing["mean_paired_correctness_difference"] == -1
    assert missing["all_question_missing_value_bounds"][0] == pytest.approx(-1 / 6)
    assert missing["all_question_missing_value_bounds"][1] == 0
    assert missing["family_bootstrap_95_interval"] is None


def test_family_bootstrap_preserves_per_question_weight_with_unequal_family_sizes():
    cases = [SimpleNamespace(id=str(i), family="large" if i < 3 else "small") for i in range(4)]
    values = {(c.id, v, r): (1.0 if c.family == "large" and v == "multi_agent_rag" else 0.0)
              for c in cases for v in ("multi_agent_rag", "single_agent_rag") for r in range(3)}
    result = primary_comparison(cases, values, replicates=100, seed=2)
    assert result["complete_pairs"] == 4 and result["mean_paired_correctness_difference"] == 0.75
    assert result["by_family"]["large"] == {"complete_pairs": 3, "mean_difference": 1}
    assert result["by_family"]["small"] == {"complete_pairs": 1, "mean_difference": 0}


def test_shared_judge_prompt_retains_development_payload_byte_for_byte():
    """Refactoring the rubric must not invalidate earlier calibration inputs."""
    from hashlib import sha256
    from app.evaluation.development import development_prompt, load_scoring_cases
    from app.evaluation.trace import EvaluationTrace
    root = Path("benchmarks/real-v1")
    cases = load_scoring_cases(root)
    trace = EvaluationTrace.model_validate_json(Path("docs/evaluation/final-dev/traces.jsonl").read_text("utf-8").splitlines()[0])
    # Recorded from 1b8cb5d before refactoring; works in shallow CI checkouts.
    assert sha256(development_prompt(cases[trace.case_id], trace, root).encode()).hexdigest() == \
        "ad52c119df4bf1c3d746474ebce86cfdd5ddd640aa1187b69be4c3b6ded8dc70"


def test_analysis_never_creates_an_empty_replacement_for_a_lost_ledger(tmp_path):
    from app.evaluation.test_analysis import _receipts
    with pytest.raises(ValueError, match="ledger is missing"):
        _receipts(tmp_path / "lost.sqlite")
    assert not (tmp_path / "lost.sqlite").exists()


@pytest.fixture
def judge_subset(tmp_path, monkeypatch):
    from hashlib import sha256
    from app.evaluation import test_judging
    entry = tmp_path / "cli.mjs"
    entry.write_text("fake CLI", encoding="utf-8")
    record = {"freeze_id": "a" * 64, "judge_runtime": {"cli_sha256": sha256(entry.read_bytes()).hexdigest()}}
    monkeypatch.setattr(test_judging, "validate_freeze", lambda *a: record)
    monkeypatch.setattr(test_judging, "_entry_path", lambda: entry)
    trace = EvaluationTrace.failed("REAL-TEST-001", WorkflowVariant.BASELINE_LLM, "workflow_failed")
    state = SimpleNamespace(success=False)
    def report(record, root, repeat):
        value = trace.model_copy(deep=True)
        if state.success and repeat == 1:
            value.status, value.answer = "success", "domain fact"
        return SimpleNamespace(traces=[value])
    monkeypatch.setattr(test_judging, "load_repeat", report)
    monkeypatch.setattr(test_judging, "run_task", lambda *a, **k: pytest.fail("unexpected paid judge call"))
    return test_judging, state


def test_judging_another_repeat_preserves_previous_summary_and_separate_ledgers(tmp_path, judge_subset):
    judging, _ = judge_subset
    judging.judge_repeats("unused", tmp_path, [0])
    judging.judge_repeats("unused", tmp_path, [1])
    summary = json.loads((tmp_path / "judgments/summary.json").read_text("utf-8"))
    assert summary["repeats"] == [0, 1] and len(summary["units"]) == 2
    assert (tmp_path / "judgments/repeat-0.sqlite").exists()
    assert (tmp_path / "judgments/repeat-1.sqlite").exists()
    assert not (tmp_path / "judgments/ledger.sqlite").exists()


def test_invalid_cached_judgment_stops_before_any_new_paid_jobs(tmp_path, judge_subset):
    judging, state = judge_subset
    judging.judge_repeats("unused", tmp_path, [0])
    path = tmp_path / "judgments/r0-answer-001.json"
    entry = json.loads(path.read_text("utf-8"))
    entry["identity"]["trace_sha256"] = "b" * 64
    path.write_text(json.dumps(entry), encoding="utf-8")
    state.success = True
    with pytest.raises(ValueError, match="saved judgment identity"):
        judging.judge_repeats("unused", tmp_path, [1])
