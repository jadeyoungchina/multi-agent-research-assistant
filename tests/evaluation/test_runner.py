from datetime import datetime, timezone
from pathlib import Path
import subprocess

import pytest

from app.config import Settings
from app.evaluation.dataset import load_benchmark_cases
from app.evaluation.models import WorkflowVariant
from app.evaluation.trace import EvaluationTrace


class RecordingWorkflow:
    def __init__(self, variant, calls, fail_case_id=None, mutate_sources=False):
        self.variant = variant
        self.calls = calls
        self.fail_case_id = fail_case_id
        self.mutate_sources = mutate_sources
        self.last_trace = None

    def run(self, case_id, question, source_files):
        self.calls.append((case_id, self.variant, question, list(source_files)))
        if self.mutate_sources:
            source_files.clear()
        if case_id == self.fail_case_id:
            raise RuntimeError("Authorization: Bearer sk-do-not-record")
        self.last_trace = EvaluationTrace(
            case_id=case_id, variant=self.variant, status="success", answer="12 MW solar array.",
            latency_ms=999999, prompt_tokens=10, completion_tokens=3, total_tokens=13,
            model_calls=2, retry_count=1, critic_loops=0, provider="fake", model="fake-chat",
        )
        return self.last_trace


@pytest.fixture
def cases():
    return load_benchmark_cases(Path("benchmarks/cases.jsonl"))[:2]


@pytest.fixture
def workflows():
    calls = []
    return {variant: RecordingWorkflow(variant, calls) for variant in reversed(WorkflowVariant)}


def test_runner_produces_sorted_identical_case_variant_matrix(cases, workflows):
    from app.evaluation.runner import evaluate_benchmark

    workflows[WorkflowVariant.BASELINE_LLM].mutate_sources = True
    report = evaluate_benchmark(list(reversed(cases)), workflows)
    expected = [(case_id, variant) for case_id in ("BENCH-001", "BENCH-002") for variant in WorkflowVariant]
    assert [(trace.case_id, trace.variant) for trace in report.traces] == expected
    calls = workflows[WorkflowVariant.BASELINE_LLM].calls
    assert [(call[0], call[1]) for call in calls] == expected
    assert all(call[3] == ["solar-storage.md"] for call in calls)
    assert all(call[2] == cases[index // 4].question for index, call in enumerate(calls))
    assert all(case.source_files == ["solar-storage.md"] for case in cases)
    assert len(report.scores) == 8
    assert report.aggregate.score_count == 8
    assert report.aggregate.mean_retry_count == 1.0
    assert report.aggregate.mean_total_tokens == 13.0
    assert list(report.by_variant) == list(WorkflowVariant)
    assert all(metrics.score_count == 2 for metrics in report.by_variant.values())


def test_one_failure_does_not_abort_remaining_runs(cases, workflows):
    from app.evaluation.runner import evaluate_benchmark

    workflows[WorkflowVariant.LLM_RAG].fail_case_id = cases[0].id
    report = evaluate_benchmark(cases, workflows)
    assert len(report.traces) == 8
    assert report.aggregate.failure_count == 1
    assert report.aggregate.failure_rate == 0.125
    failed = report.traces[1]
    assert failed.status == "failed" and failed.error_code == "workflow_failed"
    assert failed.provider == "fake" and failed.model == "fake-chat"
    assert "sk-do-not-record" not in report.model_dump_json()
    assert report.traces[-1].status == "success"


def test_external_timing_overwrites_self_report_and_preserves_adapter_trace(cases, workflows, monkeypatch):
    from app.evaluation import runner

    ticks = iter([1.0, 1.125, 3.0, 3.375])
    monkeypatch.setattr(runner, "perf_counter", lambda: next(ticks))
    workflow = workflows[WorkflowVariant.BASELINE_LLM]
    report = runner.evaluate_benchmark(cases, {WorkflowVariant.BASELINE_LLM: workflow})
    assert [trace.latency_ms for trace in report.traces] == [125, 375]
    assert [score.latency_ms for score in report.scores] == [125, 375]
    assert report.aggregate.p50_latency_ms == 125
    assert report.aggregate.p95_latency_ms == 375
    assert workflow.last_trace.latency_ms == 999999


def test_failure_latency_is_measured_externally(cases, workflows, monkeypatch):
    from app.evaluation import runner

    ticks = iter([1.0, 1.250])
    monkeypatch.setattr(runner, "perf_counter", lambda: next(ticks))
    workflow = workflows[WorkflowVariant.BASELINE_LLM]
    workflow.fail_case_id = cases[0].id
    report = runner.evaluate_benchmark(cases[:1], {WorkflowVariant.BASELINE_LLM: workflow})
    assert report.traces[0].latency_ms == report.scores[0].latency_ms == 250


def test_mismatched_adapter_trace_is_isolated(cases, workflows):
    from app.evaluation.runner import evaluate_benchmark

    class WrongIdentity(RecordingWorkflow):
        def run(self, *args):
            return super().run(*args).model_copy(update={"case_id": "BENCH-999"})

    workflows[WorkflowVariant.LLM_RAG] = WrongIdentity(WorkflowVariant.LLM_RAG, [])
    report = evaluate_benchmark(cases, workflows)
    assert len(report.traces) == 8
    assert report.aggregate.failure_count == 2
    assert report.traces[1].case_id == "BENCH-001"
    assert report.traces[1].error_code == "invalid_trace"


def test_unknown_error_codes_from_returned_traces_are_sanitized(cases):
    from app.evaluation.runner import evaluate_benchmark

    class UnsafeFailure:
        def run(self, case_id, question, source_files):
            return EvaluationTrace.failed(case_id, WorkflowVariant.BASELINE_LLM, "Authorization: Bearer sk-do-not-record")

    report = evaluate_benchmark(cases, {WorkflowVariant.BASELINE_LLM: UnsafeFailure()})
    assert all(trace.error_code == "workflow_failed" for trace in report.traces)
    assert "sk-do-not-record" not in report.model_dump_json()


def test_corpus_digest_depends_on_relative_names_but_not_root(cases, workflows, tmp_path):
    from app.evaluation.runner import corpus_sha256

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.md").write_text("same bytes", encoding="utf-8")
    (second / "a.md").write_text("same bytes", encoding="utf-8")
    assert corpus_sha256(first) == corpus_sha256(second)
    (second / "a.md").rename(second / "b.md")
    assert corpus_sha256(first) != corpus_sha256(second)


def test_metadata_records_utc_git_hashes_and_allowlisted_configuration(cases, workflows, tmp_path):
    from app.evaluation.runner import EvaluationReport, evaluate_benchmark

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "solar-storage.md").write_text("12 MW solar array", encoding="utf-8")
    settings = Settings(
        _env_file=None, chat_provider="openai", embedding_provider="fake",
        openai_api_key="sk-do-not-record", openai_base_url="https://user:password@host/api?key=secret",
        openai_chat_model="safe-model",
    )
    started = datetime.now(timezone.utc)
    report = evaluate_benchmark(cases, workflows, corpus_dir=corpus, settings=settings)
    assert started <= report.metadata.run_at_utc <= datetime.now(timezone.utc)
    assert report.metadata.run_at_utc.utcoffset().total_seconds() == 0
    assert report.metadata.git_commit == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert len(report.metadata.corpus_sha256) == 64
    assert report.metadata.prompt_version
    assert report.metadata.configuration.provider == "openai"
    assert report.metadata.configuration.model == "safe-model"
    encoded = report.model_dump_json()
    assert not any(secret in encoded for secret in ("sk-do-not-record", "password", "base_url", "api_key"))
    assert EvaluationReport.model_validate_json(encoded) == report
    other = evaluate_benchmark(cases, workflows, corpus_dir=corpus, settings=settings)
    assert other.metadata.corpus_sha256 == report.metadata.corpus_sha256
    (corpus / "solar-storage.md").write_text("changed bytes", encoding="utf-8")
    changed = evaluate_benchmark(cases, workflows, corpus_dir=corpus, settings=settings)
    assert changed.metadata.corpus_sha256 != report.metadata.corpus_sha256


@pytest.mark.parametrize("bad_label", ["Bearer sk-do-not-record", "https://user:secret@host", "sk-do-not-record"])
def test_unsafe_configured_model_labels_are_not_serialized(cases, workflows, bad_label):
    from app.evaluation.runner import evaluate_benchmark

    settings = Settings(_env_file=None, chat_provider="openai", openai_chat_model=bad_label)
    report = evaluate_benchmark(cases, workflows, settings=settings)
    assert bad_label not in report.model_dump_json()


@pytest.mark.parametrize("empty_cases, empty_workflows", [(True, False), (False, True)])
def test_empty_benchmark_is_rejected(cases, workflows, empty_cases, empty_workflows):
    from app.evaluation.runner import evaluate_benchmark

    with pytest.raises(ValueError):
        evaluate_benchmark([] if empty_cases else cases, {} if empty_workflows else workflows)
