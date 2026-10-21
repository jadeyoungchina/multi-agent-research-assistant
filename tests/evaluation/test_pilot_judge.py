import importlib.util
from pathlib import Path

import pytest


def _script():
    spec = importlib.util.spec_from_file_location("judge_dev_pilot", Path("scripts/judge_dev_pilot.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_or_truncated_judge_output_is_rejected():
    decode = _script().decode_judgment
    for text in ['{"rubric_version":"real-semantic-v1"}', '```json\n{"claims":[]', 'prefix {} suffix']:
        with pytest.raises(ValueError):
            decode(text)


def test_held_out_trace_stops_before_paid_task(tmp_path, monkeypatch):
    from app.evaluation.runner import evaluate_benchmark
    from app.evaluation.real_dataset import validate_real_benchmark
    from app.evaluation.models import WorkflowVariant
    from app.evaluation.trace import EvaluationTrace
    script = _script()
    dev, _ = validate_real_benchmark(Path("benchmarks/real-v1"))
    class Workflow:
        def run(self, case_id, *args):
            return EvaluationTrace.failed(case_id, WorkflowVariant.BASELINE_LLM, "provider_timeout")
    report = evaluate_benchmark(dev.cases[:1], {WorkflowVariant.BASELINE_LLM: Workflow()},
                                candidate_source_files=dev.candidate_source_files,
                                corpus_dir=Path("benchmarks/real-v1/corpus"), dataset_path=Path("benchmarks/real-v1/dev.jsonl"))
    report.traces[0].case_id = "REAL-TEST-001"
    (tmp_path / "results.json").write_text(report.model_dump_json(), "utf-8")
    monkeypatch.setattr(script, "run_task", lambda *a, **k: pytest.fail("test set must not reach paid judge"))
    assert script.main(["--run-dir", str(tmp_path), "--ledger", str(tmp_path / "ledger.db"),
                        "--output-dir", str(tmp_path / "judge"), "--execute"]) == 2
