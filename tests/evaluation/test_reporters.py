import csv
import json
from pathlib import Path

import pytest

from app.evaluation.dataset import load_benchmark_cases
from app.evaluation.models import WorkflowVariant
from app.evaluation.runner import evaluate_benchmark
from app.evaluation.trace import EvaluationTrace, EvidenceSnapshot


@pytest.fixture
def report():
    class Workflow:
        def __init__(self, variant):
            self.variant = variant

        def run(self, case_id, question, source_files):
            if self.variant == WorkflowVariant.BASELINE_LLM:
                return EvaluationTrace.failed(case_id, self.variant, "provider_timeout")
            return EvaluationTrace(
                case_id=case_id, variant=self.variant, status="success",
                answer='The pilot used a 12 MW solar array.\nA "quoted", second line.',
                retrieved_evidence=[EvidenceSnapshot(
                    id="e1", source_file="solar-storage.md", page_number=None,
                    chunk_index=0, text="12 MW solar array",
                )], cited_evidence_ids=["e1"],
                latency_ms=0, prompt_tokens=8, completion_tokens=4, total_tokens=12,
                model_calls=2, retry_count=1, critic_loops=0, provider="fake", model="fake-chat",
            )

    cases = load_benchmark_cases(Path("benchmarks/cases.jsonl"))[:1]
    return evaluate_benchmark(cases, {variant: Workflow(variant) for variant in WorkflowVariant})


def test_reporters_write_typed_json_and_lossless_csv_rows(tmp_path, report):
    from app.evaluation.reporters import write_reports
    from app.evaluation.runner import EvaluationReport

    paths = write_reports(report, tmp_path)
    assert {path.name for path in paths} == {"results.json", "results.csv", "report.md"}
    saved = EvaluationReport.model_validate_json((tmp_path / "results.json").read_text("utf-8"))
    assert saved == report
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert [row["variant"] for row in rows] == [variant.value for variant in WorkflowVariant]
    assert all(row["case_id"] == "BENCH-001" for row in rows)
    assert rows[1]["answer"] == report.traces[1].answer
    assert rows[0]["error_code"] == "provider_timeout"
    assert rows[1]["retry_count"] == "1"
    assert rows[1]["total_tokens"] == "12"
    assert rows[1]["citation_precision"] == "1.0"


def test_markdown_contains_comparison_failures_configuration_and_reproducibility(tmp_path, report):
    from app.evaluation.reporters import write_reports

    write_reports(report, tmp_path / "nested")
    markdown = (tmp_path / "nested/report.md").read_text("utf-8")
    assert all(variant.value in markdown for variant in WorkflowVariant)
    assert "provider_timeout" in markdown and "BENCH-001" in markdown
    assert report.metadata.corpus_sha256 in markdown
    assert report.metadata.git_commit in markdown
    assert report.metadata.prompt_version in markdown
    assert "fake-chat" in markdown
    assert "p50" in markdown and "p95" in markdown
    assert "single run does not establish a general conclusion" in markdown
    assert "synthetic" in markdown.casefold()
    assert "zero" in markdown.casefold() and "token" in markdown.casefold()


def test_gate_only_multi_agent_is_release_blocking(report, tmp_path):
    from app.evaluation.quality_gate import enforce_quality_gate

    payload = json.loads(Path("benchmarks/quality-gates.json").read_text("utf-8"))
    payload["baseline_llm"] = payload["multi_agent_rag"]
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps(payload), encoding="utf-8")
    assert enforce_quality_gate(report, gates) is None


def test_gate_collects_every_failed_threshold(report):
    from app.evaluation.quality_gate import QualityGateError, enforce_quality_gate

    variant = WorkflowVariant.MULTI_AGENT_RAG
    report.by_variant[variant] = report.by_variant[variant].model_copy(update={
        "success_rate": 0.5, "failure_rate": 0.5,
        "mean_retrieval_recall_at_5": 0.0, "mean_citation_precision": 0.0,
        "mean_evidence_coverage": 0.0, "mean_answer_key_point_coverage": 0.0,
    })
    with pytest.raises(QualityGateError) as caught:
        enforce_quality_gate(report)
    assert len(caught.value.failures) == 6
    for threshold in json.loads(Path("benchmarks/quality-gates.json").read_text("utf-8"))["multi_agent_rag"]:
        assert threshold in str(caught.value)


def test_gate_fails_when_release_variant_was_not_evaluated(report):
    from app.evaluation.quality_gate import QualityGateError, enforce_quality_gate

    del report.by_variant[WorkflowVariant.MULTI_AGENT_RAG]
    with pytest.raises(QualityGateError, match="multi_agent_rag"):
        enforce_quality_gate(report)


@pytest.mark.parametrize("payload", [{}, {"multi_agent_rag": {}}, {"multi_agent_rag": {"typo": 1}}])
def test_gate_rejects_missing_or_invalid_threshold_configuration(report, tmp_path, payload):
    from app.evaluation.quality_gate import enforce_quality_gate

    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        enforce_quality_gate(report, gates)
