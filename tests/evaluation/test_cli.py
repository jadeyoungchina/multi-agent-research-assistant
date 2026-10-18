import csv
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("default evaluation must not make network requests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_fake_cli_produces_120_traces_three_files_and_passing_gate(tmp_path, capsys, monkeypatch):
    from app.evaluation.cli import main
    from app.evaluation.runner import EvaluationReport

    monkeypatch.setenv("CHAT_PROVIDER", "dashscope")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    output = tmp_path / "results"
    assert main([
        "--dataset", "benchmarks/cases.jsonl", "--corpus", "benchmarks/corpus",
        "--provider", "fake", "--variants", "all", "--output-dir", str(output), "--enforce-gate",
    ]) == 0
    assert {path.name for path in output.iterdir()} == {"results.json", "results.csv", "report.md"}
    report = EvaluationReport.model_validate_json((output / "results.json").read_text("utf-8"))
    assert len(report.traces) == len(report.scores) == 120
    assert report.aggregate.success_rate == 1.0
    multi = report.by_variant["multi_agent_rag"]
    assert multi.mean_citation_precision == multi.mean_evidence_coverage == 1.0
    assert multi.mean_answer_key_point_coverage >= 0.8
    assert len(report.metadata.dataset_sha256) == 64
    assert report.metadata.configuration.provider == report.metadata.configuration.embedding_provider == "fake"
    assert report.metadata.configuration.retrieval_top_k == 5
    assert all(trace.model_calls > 0 for trace in report.traces)
    assert all(trace.critic_loops <= 2 for trace in report.traces)
    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 120
    console = capsys.readouterr()
    assert "120 traces" in console.out and "gate PASS" in console.out
    assert not console.err


@pytest.mark.parametrize("contents", ["not-json", "", "{}"])
def test_invalid_dataset_returns_two_without_outputs(tmp_path, contents, capsys):
    from app.evaluation.cli import main

    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(contents, encoding="utf-8")
    output = tmp_path / "results"
    assert main(["--dataset", str(dataset), "--output-dir", str(output)]) == 2
    assert not output.exists()
    assert "invalid dataset" in capsys.readouterr().err.casefold()


def test_missing_corpus_returns_invalid_dataset(tmp_path):
    from app.evaluation.cli import main

    assert main(["--corpus", str(tmp_path / "missing")]) == 2


@pytest.mark.parametrize("kind", [
    "absolute_inside_root", "unsupported_suffix", "same_content_alias",
    "normalized_path_alias", "malformed_pdf", "oversized",
])
def test_invalid_sources_return_two_before_provider_setup(tmp_path, capsys, monkeypatch, kind):
    from app.evaluation import cli

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / ("source.exe" if kind == "unsupported_suffix" else "source.pdf" if kind == "malformed_pdf" else "source.md")
    source.write_text("expected phrase", encoding="utf-8")
    if kind == "oversized":
        monkeypatch.setenv("MAX_UPLOAD_FILE_BYTES", "14")
    source_files = [str(source.resolve()) if kind == "absolute_inside_root" else source.name]
    if kind == "same_content_alias":
        (corpus / "alias.md").write_bytes(source.read_bytes())
        source_files.append("alias.md")
    elif kind == "normalized_path_alias":
        source_files.append("./source.md")
    payload = {
        "id": "BENCH-001", "question": "What does the source say?", "source_files": source_files,
        "expected_evidence": [{"source_file": source_files[0], "contains": "expected phrase"}],
        "answer_key_points": ["Expected phrase."],
    }
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(payload), encoding="utf-8")
    provider_setup_calls = []
    build_provider = cli.build_demo_fake_provider

    def record_provider_setup():
        provider_setup_calls.append(True)
        return build_provider()

    monkeypatch.setattr(cli, "build_demo_fake_provider", record_provider_setup)
    output = tmp_path / "results"
    assert cli.main([
        "--dataset", str(dataset), "--corpus", str(corpus), "--provider", "fake",
        "--output-dir", str(output),
    ]) == 2
    assert provider_setup_calls == []
    assert not output.exists()
    assert "invalid dataset" in capsys.readouterr().err.casefold()


def test_source_exactly_at_configured_size_limit_is_accepted(tmp_path, monkeypatch):
    from app.evaluation import cli

    monkeypatch.setenv("MAX_UPLOAD_FILE_BYTES", "15")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "source.md").write_bytes(b"expected phrase")
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({
        "id": "BENCH-001", "question": "What does the source say?", "source_files": ["source.md"],
        "expected_evidence": [{"source_file": "source.md", "contains": "expected phrase"}],
        "answer_key_points": ["Expected phrase."],
    }), encoding="utf-8")
    output = tmp_path / "results"
    assert cli.main([
        "--dataset", str(dataset), "--corpus", str(corpus), "--provider", "fake",
        "--variants", "baseline_llm", "--output-dir", str(output),
    ]) == 0
    assert len(json.loads((output / "results.json").read_text("utf-8"))["traces"]) == 1


def test_provider_setup_value_error_remains_execution_error(tmp_path, monkeypatch, capsys):
    from app.evaluation import cli

    def failing_provider():
        raise ValueError("provider could not initialize")

    monkeypatch.setattr(cli, "build_demo_fake_provider", failing_provider)
    assert cli.main(["--provider", "fake", "--output-dir", str(tmp_path / "results")]) == 4
    assert "execution error" in capsys.readouterr().err.casefold()


def test_gate_failure_returns_three_and_preserves_reports(tmp_path, capsys):
    from app.evaluation.cli import main

    dataset = tmp_path / "cases.jsonl"
    payload = json.loads(Path("benchmarks/cases.jsonl").read_text("utf-8").splitlines()[0])
    payload["answer_key_points"] = ["Gold-only private sentence never supplied to the workflow."]
    dataset.write_text(json.dumps(payload), encoding="utf-8")
    assert main([
        "--dataset", str(dataset), "--provider", "fake", "--variants", "multi_agent_rag",
        "--output-dir", str(tmp_path / "results"), "--enforce-gate",
    ]) == 3
    assert len(list((tmp_path / "results").iterdir())) == 3
    assert "minimum_answer_key_point_coverage" in capsys.readouterr().err
    report = json.loads((tmp_path / "results/results.json").read_text("utf-8"))
    assert "Gold-only" not in report["traces"][0]["answer"]


def test_execution_error_returns_four_without_secret_details(tmp_path, capsys, monkeypatch):
    from app.evaluation import cli

    def failing(*args, **kwargs):
        raise RuntimeError("Authorization: Bearer sk-do-not-record")

    monkeypatch.setattr(cli, "evaluate_benchmark", failing)
    assert cli.main(["--provider", "fake", "--output-dir", str(tmp_path)]) == 4
    error = capsys.readouterr().err
    assert "execution error" in error.casefold()
    assert "sk-do-not-record" not in error and "Authorization" not in error


def test_selected_variants_use_enum_order_and_same_cases(tmp_path):
    from app.evaluation.cli import main

    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(Path("benchmarks/cases.jsonl").read_text("utf-8").splitlines()[0], encoding="utf-8")
    assert main([
        "--dataset", str(dataset), "--variants", "single_agent_rag,baseline_llm",
        "--output-dir", str(tmp_path / "results"),
    ]) == 0
    report = json.loads((tmp_path / "results/results.json").read_text("utf-8"))
    assert [(trace["case_id"], trace["variant"]) for trace in report["traces"]] == [
        ("BENCH-001", "baseline_llm"), ("BENCH-001", "single_agent_rag"),
    ]


def test_module_entry_point_returns_dataset_exit_code(tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "app.evaluation.cli", "--dataset", str(tmp_path / "missing.jsonl"),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid dataset" in result.stderr.casefold()


def test_generated_artifacts_are_ignored():
    result = subprocess.run([
        "git", "check-ignore", "artifacts/evaluation/fake/results.json",
        "artifacts/evaluation/fake/results.csv", "artifacts/evaluation/fake/report.md",
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert len(result.stdout.splitlines()) == 3


def test_paid_resume_does_not_generate_completed_units_again(tmp_path, monkeypatch):
    from app.evaluation import cli
    from app.domain.providers import ProviderMetadata

    calls = []
    class Chat:
        def generate(self, messages):
            calls.append(messages)
            return "answer", ProviderMetadata(provider="dashscope", model="qwen3.7-flash", latency_ms=1)
    monkeypatch.setattr(cli, "build_chat_provider", lambda *args, **kwargs: Chat())
    args = ["--provider", "dashscope", "--embedding-provider", "fake", "--variants", "baseline_llm",
            "--limit", "2", "--ledger", str(tmp_path / "budget.db"), "--output-dir", str(tmp_path / "run")]
    assert cli.main(args) == 0
    first = json.loads((tmp_path / "run/results.json").read_text("utf-8"))
    assert len(calls) == 2
    assert cli.main(args + ["--resume"]) == 0
    assert len(calls) == 2
    second = json.loads((tmp_path / "run/results.json").read_text("utf-8"))
    assert first["traces"] == second["traces"]
    assert len((tmp_path / "run/traces.jsonl").read_text("utf-8").splitlines()) == 2
    assert cli.main(args + ["--resume", "--max-output-tokens", "1024"]) == 4
    assert len(calls) == 2
    assert cli.main(args + ["--resume", "--order-seed", "42"]) == 4
    assert len(calls) == 2


@pytest.mark.parametrize("selection", ["BENCH-999", "BENCH-001,BENCH-001", "BENCH-001,"])
def test_invalid_case_selection_stops_before_provider_setup(tmp_path, monkeypatch, selection):
    from app.evaluation import cli
    monkeypatch.setattr(cli, "_build_workflows", lambda *args, **kwargs: pytest.fail("invalid selection must not call providers"))
    assert cli.main(["--case-ids", selection, "--output-dir", str(tmp_path)]) == 2


def test_explicit_real_cases_keep_the_entire_split_pool(tmp_path, monkeypatch):
    from app.evaluation import cli
    pools = []
    build = cli._build_workflows
    def capture(*args, **kwargs):
        pools.append(kwargs["candidate_source_files"])
        return build(*args, **kwargs)
    monkeypatch.setattr(cli, "_build_workflows", capture)
    selected = ["REAL-DEV-001", "REAL-DEV-007", "REAL-DEV-013", "REAL-DEV-019", "REAL-DEV-020"]
    output = tmp_path / "run"
    assert cli.main(["--real-manifest", "benchmarks/real-v1/corpus-manifest.json", "--allow-ai-reviewed",
                 "--case-ids", ",".join(selected), "--variants", "llm_rag", "--order-seed", "42",
                 "--output-dir", str(output)]) == 0
    report = json.loads((output / "results.json").read_text("utf-8"))
    assert {trace["case_id"] for trace in report["traces"]} == set(selected)
    assert len(pools) == 1 and len(set(pools[0])) == 8


def test_budget_stop_saves_partial_matrix_and_ledger(tmp_path, monkeypatch):
    from app.evaluation import cli
    from app.evaluation.budget import BudgetError

    calls = []
    class Chat:
        def generate(self, messages):
            calls.append(messages)
            raise BudgetError("budget_exhausted", "stop")
    monkeypatch.setattr(cli, "build_chat_provider", lambda *args, **kwargs: Chat())
    assert cli.main([
        "--provider", "dashscope", "--embedding-provider", "fake", "--variants", "baseline_llm",
        "--limit", "2", "--ledger", str(tmp_path / "budget.db"), "--output-dir", str(tmp_path / "run"),
    ]) == 5
    report = json.loads((tmp_path / "run/results.json").read_text("utf-8"))
    assert not report["execution_complete"] and report["planned_trace_count"] == 2
    assert len(report["traces"]) == 1 and len(calls) == 1
    assert (tmp_path / "run/budget-summary.json").exists()
