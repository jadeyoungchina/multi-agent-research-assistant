import json
from pathlib import Path
import shutil

import pytest

from app.evaluation.real_dataset import validate_real_benchmark
from app.evaluation.reference_review import validate_ai_reference_review


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "benchmarks/real-v1"


def test_complete_ai_review_preserves_actual_human_status():
    report = validate_ai_reference_review(SUITE)
    assert len(report["reviews"]) == report["accepted_count"] == 60
    assert all(item["deepseek_tasks"] and item["codex_adjudication"] for item in report["reviews"])
    assert all(case.human_review.status == "pending" for suite in validate_real_benchmark(SUITE) for case in suite.cases)


@pytest.fixture
def copied(tmp_path):
    target = tmp_path / "real-v1"
    shutil.copytree(SUITE, target)
    return target


@pytest.mark.parametrize("kind", ["missing_case", "stale_case", "unresolved", "missing_pool", "answer_found"])
def test_incomplete_or_stale_ai_review_is_rejected(copied, kind):
    path = copied / "review/final-audit.json"
    report = json.loads(path.read_text("utf-8"))
    if kind == "missing_case":
        report["reviews"].pop()
    elif kind == "stale_case":
        report["reviews"][0]["reference_sha256"] = "0" * 64
    elif kind == "unresolved":
        report["reviews"][0]["status"] = "unresolved"
    elif kind == "missing_pool":
        report["absence_audit"]["dev"]["covered_sources"].pop()
    else:
        report["absence_audit"]["dev"]["conclusions"]["REAL-DEV-005"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_ai_reference_review(copied)


def test_changed_reference_invalidates_ai_pilot_before_provider_setup(copied, monkeypatch, tmp_path):
    from app.evaluation import cli
    path = copied / "dev.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows[0]["question"] += " Changed reference."
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(cli, "_build_workflows", lambda *args, **kwargs: pytest.fail("stale AI audit must stop paid setup"))
    assert cli.main(["--provider", "dashscope", "--real-manifest", str(copied / "corpus-manifest.json"),
                     "--allow-ai-reviewed", "--output-dir", str(tmp_path / "run")]) == 2


def test_later_human_review_does_not_invalidate_unchanged_ai_reference(copied):
    path = copied / "dev.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows[0]["human_review"] = dict(status="approved", reviewer="Test-only synthetic reviewer",
                                    reviewed_at="2026-10-18T15:00:00+08:00", notes="test fixture only")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    assert validate_ai_reference_review(copied)["accepted_count"] == 60


def test_ai_reviewed_run_labels_ai_and_human_review_separately(tmp_path):
    from app.evaluation.cli import main
    output = tmp_path / "run"
    assert main(["--provider", "fake", "--real-manifest", str(SUITE / "corpus-manifest.json"),
                 "--allow-ai-reviewed", "--limit", "1", "--variants", "llm_rag", "--output-dir", str(output)]) == 0
    metadata = json.loads((output / "results.json").read_text("utf-8"))["metadata"]
    assert metadata["references_ai_reviewed"] is True
    assert metadata["references_human_reviewed"] is False
    assert len(metadata["ai_reference_review_sha256"]) == 64


def test_all_gold_units_are_reachable_after_actual_document_normalization_and_chunking():
    from app.config import Settings
    from app.retrieval.chunking import chunk_pages
    from app.retrieval.loaders import load_document
    settings = Settings(_env_file=None)
    for suite in validate_real_benchmark(SUITE):
        chunks = {name: chunk_pages("audit", name, load_document(src.content, name, src.media_type),
                                    settings.chunk_size, settings.chunk_overlap) for name, src in suite.sources.items()}
        for case in suite.cases:
            for evidence in case.expected_evidence:
                assert any(evidence.contains.casefold() in chunk.content.casefold() for chunk in chunks[evidence.source_file]), case.id
