import json
from pathlib import Path

import pytest

from app.evaluation.dataset import load_benchmark_cases, validate_benchmark_corpus


DATASET_PATH = Path("benchmarks/cases.jsonl")
CORPUS_PATH = Path("benchmarks/corpus")

_VALID_CASE_FIELDS: dict[str, object] = {
    "question": "What does the synthetic source state?",
    "source_files": ["source.md"],
    "expected_evidence": [{"source_file": "source.md", "contains": "expected phrase"}],
    "answer_key_points": ["Expected phrase."],
    "max_critic_loops": 2,
    "latency_budget_ms": 5000,
}


def test_default_dataset_has_thirty_sequential_unique_cases() -> None:
    """Removing or duplicating a case must invalidate the release benchmark."""
    cases = load_benchmark_cases(DATASET_PATH)

    assert len(cases) == 30
    assert [case.id for case in cases] == [f"BENCH-{number:03d}" for number in range(1, 31)]


def test_expected_phrases_exist_in_declared_sources() -> None:
    """Evidence expectations must remain grounded in the declared synthetic corpus."""
    cases = load_benchmark_cases(DATASET_PATH)

    validate_benchmark_corpus(cases, CORPUS_PATH)


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {"id": "BENCH-001", **_VALID_CASE_FIELDS},
                {"id": "BENCH-001", **_VALID_CASE_FIELDS},
            ],
            "sequential",
        ),
        (
            [{"id": "BENCH-002", **_VALID_CASE_FIELDS}],
            "sequential",
        ),
    ],
)
def test_loader_rejects_duplicate_or_non_sequential_ids(
    tmp_path: Path, records: list[dict[str, object]], message: str
) -> None:
    """A malformed ID sequence must not silently change the evaluated population."""
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_benchmark_cases(dataset_path)


def test_corpus_validation_rejects_path_escape(tmp_path: Path) -> None:
    """A source path outside the corpus cannot be read as benchmark evidence."""
    case_path = tmp_path / "case.jsonl"
    case_path.write_text(
        json.dumps({"id": "BENCH-001", **_VALID_CASE_FIELDS, "source_files": ["../secret.md"]}),
        encoding="utf-8",
    )
    cases = load_benchmark_cases(case_path)

    with pytest.raises(ValueError, match="parent|escapes"):
        validate_benchmark_corpus(cases, tmp_path / "corpus")


def test_corpus_validation_rejects_parent_component_inside_corpus(tmp_path: Path) -> None:
    """Parent path components are invalid even when normalization stays in the corpus."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "solar-storage.md").write_text("expected phrase", encoding="utf-8")
    source_file = "subdir/../solar-storage.md"
    case_path = tmp_path / "case.jsonl"
    case_path.write_text(
        json.dumps(
            {
                "id": "BENCH-001",
                **_VALID_CASE_FIELDS,
                "source_files": [source_file],
                "expected_evidence": [{"source_file": source_file, "contains": "expected phrase"}],
            }
        ),
        encoding="utf-8",
    )
    cases = load_benchmark_cases(case_path)

    with pytest.raises(ValueError, match="parent"):
        validate_benchmark_corpus(cases, corpus_dir)


def test_corpus_validation_rejects_missing_source(tmp_path: Path) -> None:
    """Expected evidence must name a source file that exists in the corpus."""
    case_path = tmp_path / "case.jsonl"
    case_path.write_text(json.dumps({"id": "BENCH-001", **_VALID_CASE_FIELDS}), encoding="utf-8")
    cases = load_benchmark_cases(case_path)

    with pytest.raises(ValueError, match="does not exist"):
        validate_benchmark_corpus(cases, tmp_path / "corpus")


def test_loader_rejects_blank_answer_key_points(tmp_path: Path) -> None:
    """Evaluation scoring data must not use blank answer points."""
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "BENCH-001", **_VALID_CASE_FIELDS, "answer_key_points": [""]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="answer_key_points"):
        load_benchmark_cases(dataset_path)


def test_loader_rejects_excess_critic_loops(tmp_path: Path) -> None:
    """Evaluation cases must not permit more than two Critic loops."""
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "BENCH-001", **_VALID_CASE_FIELDS, "max_critic_loops": 3}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_critic_loops"):
        load_benchmark_cases(dataset_path)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("unexpected_evaluator_field", True, "extra_forbidden"),
        ("max_critic_loops", "2", "int_type"),
    ],
)
def test_loader_rejects_unknown_fields_and_coerced_types(
    tmp_path: Path, field_name: str, value: object, message: str
) -> None:
    """Strict evaluator records reject undeclared fields and type coercion."""
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "BENCH-001", **_VALID_CASE_FIELDS, field_name: value}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_benchmark_cases(dataset_path)
