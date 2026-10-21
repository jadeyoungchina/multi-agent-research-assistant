from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evaluation import frozen, runner
from app.evaluation.budget import BudgetLedger, PRICES
from app.evaluation.models import WorkflowVariant
from app.evaluation.trace import EvaluationTrace


@pytest.fixture
def record():
    design = json.loads(Path("benchmarks/real-v1/test-preparation.json").read_text("utf-8"))
    return {"freeze_id": "a" * 64, "design": design, "embedding_batch_size": 10,
            "prompt_version": "evaluation-v8", "judge_runtime": {"cli_sha256": "b" * 64}}


@pytest.fixture
def paid_simulation(monkeypatch, record):
    """Real ledger/journal/runner; only the provider-facing workflow is replaced."""
    monkeypatch.setattr(frozen, "validate_freeze", lambda *a, **k: record)
    actual_output = runner.subprocess.check_output

    def clean_metadata(arguments, **kwargs):
        if arguments == ["git", "status", "--porcelain"]:
            return ""
        return actual_output(arguments, **kwargs)

    monkeypatch.setattr(runner.subprocess, "check_output", clean_metadata)
    state = SimpleNamespace(calls=[], setups=0, interrupt_at=None, setup_error=False)

    def receipt(ledger, kind, role):
        model = "text-embedding-v4" if kind == "embedding" else "qwen3.7-flash-2026-07-15"
        identifier = ledger.reserve(provider="dashscope", model=model, kind=kind, role=role,
            request_hash=sha256(str(len(ledger.summary()["attempts"])).encode()).hexdigest(),
            price=PRICES[("dashscope", model)], max_input=10, max_output=10 if kind == "chat" else 0)
        return identifier

    def build(settings, cases, corpus, variants, resources, ledger, *, setup_ledger, candidate_source_files, **kwargs):
        state.setups += 1
        index_id = receipt(setup_ledger, "embedding", "corpus_index")
        setup_ledger.settle(index_id, input_tokens=1, output_tokens=0)
        if state.setup_error:
            raise RuntimeError("simulated setup failure")
        assert len(candidate_source_files) == 16
        configuration = runner.ProviderConfiguration.from_settings(settings)

        class Workflow:
            def __init__(self, variant):
                self.variant = variant

            def run(self, case_id, question, sources):
                assert type(case_id) is str and type(question) is str
                assert sources == candidate_source_files
                state.calls.append((case_id, self.variant.value))
                identifier = receipt(ledger, "chat", "evaluation")
                if state.interrupt_at == len(state.calls):
                    raise KeyboardInterrupt()
                ledger.settle(identifier, input_tokens=1, output_tokens=1)
                return EvaluationTrace(case_id=case_id, variant=self.variant, status="success", answer="simulated answer",
                    latency_ms=0, prompt_tokens=1, completion_tokens=1, total_tokens=2, model_calls=1,
                    attempt_ids=[identifier], critic_loops=0, provider=configuration.provider, model=configuration.model)
        return {variant: Workflow(variant) for variant in variants}

    monkeypatch.setattr(frozen, "_build_workflows", build)
    return state


def test_repeat_is_complete_and_completed_resume_creates_no_provider_or_setup(tmp_path, paid_simulation):
    first = frozen.run_repeat("unused", tmp_path, 0)
    calls, setups = len(paid_simulation.calls), paid_simulation.setups
    assert len(first.traces) == 160 and first.execution_complete
    assert calls == 160 and setups == 1
    second = frozen.run_repeat("unused", tmp_path, 0)
    assert second == first
    assert len(paid_simulation.calls) == calls and paid_simulation.setups == setups
    assert len(BudgetLedger(tmp_path / "repeat-0/index.sqlite").summary()["attempts"]) == 1


def test_interrupted_unit_blocks_calls_until_retained_failure_and_setup_stays_separate(tmp_path, paid_simulation):
    paid_simulation.interrupt_at = 2
    with pytest.raises(KeyboardInterrupt):
        frozen.run_repeat("unused", tmp_path, 0)
    with pytest.raises(ValueError, match="uncertain paid unit"):
        frozen.run_repeat("unused", tmp_path, 0)
    assert len(paid_simulation.calls) == 2 and paid_simulation.setups == 1
    paid_simulation.interrupt_at = None
    report = frozen.run_repeat("unused", tmp_path, 0, resolve_interrupted=True)
    assert len(report.traces) == 160 and sum(t.status == "failed" for t in report.traces) == 1
    assert len(paid_simulation.calls) == len(set(paid_simulation.calls)) == 160
    rows = BudgetLedger(tmp_path / "repeat-0/generation.sqlite").summary()["attempts"]
    setup = BudgetLedger(tmp_path / "repeat-0/index.sqlite").summary()["attempts"]
    assert len(rows) == 160 and len(setup) == 2
    assert all(r["kind"] == "chat" for r in rows) and all(r["kind"] == "embedding" for r in setup)
    failed = next(t for t in report.traces if t.status == "failed")
    assert failed.model_calls == 1 and not failed.usage_complete
    assert next(r for r in rows if r["id"] == failed.attempt_ids[0])["state"] == "reserved"
    assert report.aggregate.metric_denominators["latency_ms"] == 159
    performance = json.loads((tmp_path / "repeat-0/performance.json").read_text("utf-8"))
    assert len(performance["unknown_latency_retry_units"]) == 1
    assert performance["by_variant"]["all"]["latency_retry_denominator"] == 159


def test_independent_repeats_generate_fresh_requests_with_balanced_positions(tmp_path, paid_simulation, monkeypatch):
    identifiers = []
    for repeat in range(3):
        report = frozen.run_repeat("unused", tmp_path, repeat)
        identifiers.extend(i for trace in report.traces for i in trace.attempt_ids)
        for position in range(4):
            assert set(Counter(t.variant for t in report.traces[position::4]).values()) == {10}
    assert len(paid_simulation.calls) == 480 and paid_simulation.setups == 3
    assert len(identifiers) == len(set(identifiers)) == 480
    from app.evaluation import test_analysis
    monkey_record = frozen.validate_freeze("unused")
    monkeypatch.setattr(test_analysis, "validate_freeze", lambda *a: monkey_record)
    first = frozen.EvaluationReport.model_validate_json((tmp_path / "repeat-0/results.json").read_text("utf-8")).traces[0]
    frozen.write_json(tmp_path / "judgments/r0-answer-001.json", {"unit": "r0-answer-001",
        "case_id": first.case_id, "variant": first.variant.value, "human_reviewed": False, "status": "judge_error",
        "identity": {"freeze_id": monkey_record["freeze_id"], "trace_sha256": sha256(first.model_dump_json().encode()).hexdigest()}})
    result = test_analysis.analyze("unused", tmp_path)
    assert result["executed_units"] == 480 and result["execution_failures"] == 0
    assert len(result["successful_answers_missing_judgment"]) == 480
    assert result["primary"]["complete_pairs"] == 0
    assert result["primary"]["all_question_missing_value_bounds"] == [-1, 1]
    assert "r0-answer-001" in result["successful_answers_missing_judgment"]
    assert result["judge_cost_cny"] is None and result["missing_judge_ledgers"] == [0, 1, 2]


def test_setup_failure_receipts_survive_a_new_setup(tmp_path, paid_simulation):
    paid_simulation.setup_error = True
    with pytest.raises(RuntimeError):
        frozen.run_repeat("unused", tmp_path, 0)
    assert not paid_simulation.calls
    paid_simulation.setup_error = False
    frozen.run_repeat("unused", tmp_path, 0)
    assert len(BudgetLedger(tmp_path / "repeat-0/index.sqlite").summary()["attempts"]) == 2


def test_other_repeat_identity_cannot_reuse_paid_answers(tmp_path, paid_simulation):
    frozen.run_repeat("unused", tmp_path, 0)
    path = tmp_path / "repeat-0/repeat-manifest.json"
    data = json.loads(path.read_text("utf-8"))
    data["repeat_index"] = 1
    frozen.write_json(path, data)
    with pytest.raises(ValueError, match="repeat identity"):
        frozen.run_repeat("unused", tmp_path, 0)
    assert len(paid_simulation.calls) == 160


def test_generation_lock_prevents_two_process_owners(tmp_path):
    with frozen.exclusive_run(tmp_path / "lock"):
        with pytest.raises(ValueError, match="already running"):
            with frozen.exclusive_run(tmp_path / "lock"):
                pytest.fail("second owner entered")


def test_freeze_detects_input_change_and_uncommitted_record(tmp_path, monkeypatch):
    from app.evaluation import deepseek_cli
    entry = tmp_path / "cli.mjs"
    entry.write_text("test-only CLI", encoding="utf-8")
    monkeypatch.setattr(deepseek_cli, "_entry_path", lambda: entry)
    monkeypatch.setattr(frozen, "_git", lambda *a: "" if a[0] == "status" else "c" * 40)
    monkeypatch.setattr(frozen, "generation_compatibility", lambda: {"test_only": True})
    monkeypatch.setattr(frozen, "validate_calibration", lambda: {"test_only": True})
    inputs = [{"path": "app/example.py", "mode": "lf", "sha256": "1" * 64}]
    monkeypatch.setattr(frozen, "_inputs", lambda: deepcopy(inputs))
    path = tmp_path / "freeze.json"
    frozen.create_freeze(path)
    frozen.validate_freeze(path, require_committed=False)
    inputs[0]["sha256"] = "2" * 64
    with pytest.raises(ValueError, match="frozen code"):
        frozen.validate_freeze(path, require_committed=False)
    inputs[0]["sha256"] = "1" * 64
    monkeypatch.setattr(frozen, "_git", lambda *a: "uncommitted" if a[0] == "status" else "c" * 40)
    with pytest.raises(ValueError, match="clean committed"):
        frozen.validate_freeze(path)


def test_missing_final_report_is_rebuilt_from_completed_traces_without_requests(tmp_path, paid_simulation):
    frozen.run_repeat("unused", tmp_path, 0)
    (tmp_path / "repeat-0/results.json").unlink()
    report = frozen.run_repeat("unused", tmp_path, 0)
    assert report.execution_complete and len(paid_simulation.calls) == 160 and paid_simulation.setups == 1


def test_torn_final_trace_requires_resolution_without_duplicate_generation(tmp_path, paid_simulation):
    paid_simulation.interrupt_at = 2
    with pytest.raises(KeyboardInterrupt):
        frozen.run_repeat("unused", tmp_path, 0)
    with (tmp_path / "repeat-0/traces.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"case_id":')
    with pytest.raises(ValueError, match="uncertain paid unit"):
        frozen.run_repeat("unused", tmp_path, 0)
    assert len(paid_simulation.calls) == 2
    assert len((tmp_path / "repeat-0/traces.jsonl").read_text("utf-8").splitlines()) == 1


def test_invalid_freeze_is_rejected_before_directory_or_provider_creation(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("frozen code differs")
    monkeypatch.setattr(frozen, "validate_freeze", reject)
    monkeypatch.setattr(frozen, "_build_workflows", lambda *a, **k: pytest.fail("must not create clients"))
    with pytest.raises(ValueError, match="frozen code"):
        frozen.run_repeat("invalid", tmp_path / "new", 0)
    assert not (tmp_path / "new").exists()


def test_torn_utf8_tail_is_quarantined_and_never_resubmitted(tmp_path, paid_simulation):
    paid_simulation.interrupt_at = 2
    with pytest.raises(KeyboardInterrupt):
        frozen.run_repeat("unused", tmp_path, 0)
    fragment = b'{"answer":"\xe4\xb8'
    with (tmp_path / "repeat-0/traces.jsonl").open("ab") as stream:
        stream.write(fragment)
    with pytest.raises(ValueError, match="uncertain paid unit"):
        frozen.run_repeat("unused", tmp_path, 0)
    assert len(paid_simulation.calls) == 2
    recovery = json.loads((tmp_path / "repeat-0/torn-traces.jsonl").read_text("utf-8").splitlines()[0])
    assert bytes.fromhex(recovery["partial_bytes_hex"]) == fragment


def test_cached_report_metadata_must_match_frozen_order_before_zero_call_resume(tmp_path, paid_simulation):
    frozen.run_repeat("unused", tmp_path, 0)
    path = tmp_path / "repeat-0/results.json"
    report = json.loads(path.read_text("utf-8"))
    report["metadata"]["order_seed"] += 1
    frozen.write_json(path, report)
    with pytest.raises(ValueError, match="does not match the freeze"):
        frozen.run_repeat("unused", tmp_path, 0)
    assert len(paid_simulation.calls) == 160 and paid_simulation.setups == 1


def test_actual_workflow_builder_routes_document_and_query_embeddings_to_separate_ledgers(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from app.config import Settings
    from app.evaluation import cli
    from app.evaluation.dataset import load_benchmark_cases
    from app.providers.fake import DeterministicEmbeddingProvider
    requests = []

    class TaggedEmbeddings(DeterministicEmbeddingProvider):
        def __init__(self, ledger):
            super().__init__()
            self.ledger = ledger

        def embed_documents(self, texts):
            requests.append(self.ledger)
            return super().embed_documents(texts)

        def embed_query(self, text):
            requests.append(self.ledger)
            return super().embed_query(text)

    monkeypatch.setattr(cli, "build_embedding_provider", lambda settings, *, ledger, role: TaggedEmbeddings(ledger))
    answers, index = object(), object()
    settings = Settings(_env_file=None, chat_provider="fake", embedding_provider="fake", retrieval_min_similarity=-1,
        data_dir=tmp_path, upload_dir=tmp_path / "uploads", database_path=tmp_path / "app.sqlite")
    cases = load_benchmark_cases(Path("benchmarks/cases.jsonl"))[:1]
    with ExitStack() as resources:
        workflows = cli._build_workflows(settings, cases, Path("benchmarks/corpus"),
            [WorkflowVariant.LLM_RAG], resources, answers, setup_ledger=index)
        assert requests and all(item is index for item in requests)
        requests.clear()
        workflows[WorkflowVariant.LLM_RAG].run(cases[0].id, cases[0].question, list(cases[0].source_files))
        assert requests and all(item is answers for item in requests)
