from pathlib import Path
import json
import shutil

import pytest

from app.evaluation.models import WorkflowVariant
from app.evaluation.real_dataset import load_corpus_manifest, load_real_suite, validate_real_benchmark
from app.evaluation.runner import evaluate_benchmark
from app.evaluation.trace import EvaluationTrace


SUITE = Path("benchmarks/real-v1")


def test_real_corpus_is_pinned_and_design_is_balanced():
    dev, test = validate_real_benchmark(SUITE)
    assert (len(dev.cases), len(test.cases)) == (20, 40)
    assert (len(dev.sources), len(test.sources)) == (8, 16)
    assert len(dev.manifest.documents) == 24
    assert all(case.human_review.status == "pending" for case in [*dev.cases, *test.cases])
    assert all(not case.expected_evidence for case in [*dev.cases, *test.cases] if not case.expected_sufficiency)
    assert set(dev.sources).isdisjoint(test.sources)


def test_formal_references_require_actual_human_review():
    with pytest.raises(ValueError, match="human"):
        load_real_suite(SUITE / "corpus-manifest.json", SUITE / "dev.jsonl", "dev", require_human_review=True)


def test_runner_receives_whole_pool_even_after_case_limit():
    dev, _ = validate_real_benchmark(SUITE)
    calls = []

    class Recorder:
        def run(self, *args):
            calls.append(args)
            args[2].clear()  # One adapter cannot mutate another adapter's pool.
            return EvaluationTrace.failed(args[0], WorkflowVariant.LLM_RAG, "provider_timeout")

    class Observer:
        def run(self, *args):
            calls.append(args)
            return EvaluationTrace.failed(args[0], WorkflowVariant.SINGLE_AGENT_RAG, "provider_timeout")

    report = evaluate_benchmark(dev.cases[:1], {
        WorkflowVariant.LLM_RAG: Recorder(), WorkflowVariant.SINGLE_AGENT_RAG: Observer(),
    }, corpus_dir=SUITE / "corpus", candidate_source_files=dev.candidate_source_files,
       corpus_manifest_path=SUITE / "corpus-manifest.json")
    assert len(calls[1]) == 3
    assert set(calls[1][2]) == set(dev.sources)
    assert len(dev.sources) > len({e.source_file for e in dev.cases[0].expected_evidence})
    assert report.metadata.synthetic is False
    assert report.metadata.references_human_reviewed is False
    assert report.metadata.corpus_manifest_sha256
    with pytest.raises(ValueError, match="independent"):
        evaluate_benchmark(dev.cases[:1], {WorkflowVariant.LLM_RAG: Recorder()}, corpus_dir=SUITE / "corpus")


@pytest.fixture
def copied_suite(tmp_path):
    target = tmp_path / "real-v1"
    shutil.copytree(SUITE, target)
    return target


@pytest.mark.parametrize("kind", ["raw", "corpus", "license"])
def test_snapshot_or_license_tampering_is_rejected(copied_suite, kind):
    manifest = json.loads((copied_suite / "corpus-manifest.json").read_text("utf-8"))
    name = manifest["licenses"][0]["path"] if kind == "license" else manifest["documents"][0][f"{kind}_path"]
    with (copied_suite / name).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_corpus_manifest(copied_suite / "corpus-manifest.json")


def test_same_document_family_cannot_cross_splits(copied_suite):
    path = copied_suite / "corpus-manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest["documents"][0]["split"] = "test"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="crosses splits"):
        load_corpus_manifest(path)


@pytest.mark.parametrize("kind", ["quote", "line", "group"])
def test_gold_locations_and_cross_split_question_groups_are_validated(copied_suite, kind):
    path = copied_suite / "dev.jsonl"
    cases = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    if kind == "quote":
        cases[0]["expected_evidence"][0]["contains"] = "invented evidence"
    elif kind == "line":
        cases[0]["expected_evidence"][0]["start_line"] += 10000
    else:
        cases[0]["question_group"] = "REAL-TEST-001"
    path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_real_benchmark(copied_suite)


@pytest.mark.parametrize("split,extra", [("dev", []), ("test", ["--allow-unreviewed"])])
def test_unreviewed_paid_dev_and_unfrozen_paid_test_stop_before_provider_setup(split, extra, monkeypatch, tmp_path):
    from app.evaluation import cli
    monkeypatch.setattr(cli, "_build_workflows", lambda *args, **kwargs: pytest.fail("no paid setup before validation"))
    assert cli.main(["--real-manifest", str(SUITE / "corpus-manifest.json"), "--provider", "dashscope",
                     "--split", split, "--output-dir", str(tmp_path), *extra]) == 2
