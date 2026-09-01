"""Load and validate the local, synthetic benchmark dataset."""

import json
from pathlib import Path

from pydantic import ValidationError

from .models import BenchmarkCase


def load_benchmark_cases(dataset_path: Path) -> list[BenchmarkCase]:
    """Parse JSONL cases and require IDs exactly BENCH-001 onward."""
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at line {line_number}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"JSON object required at line {line_number}")
        try:
            cases.append(BenchmarkCase.model_validate(payload))
        except ValidationError as error:
            raise ValueError(f"invalid benchmark case at line {line_number}: {error}") from error

    expected_ids = [f"BENCH-{number:03d}" for number in range(1, len(cases) + 1)]
    actual_ids = [case.id for case in cases]
    if actual_ids != expected_ids:
        raise ValueError("benchmark case IDs must be unique and sequential from BENCH-001")
    return cases


def validate_benchmark_corpus(cases: list[BenchmarkCase], corpus_dir: Path) -> None:
    """Ensure all local evidence files and expected phrases stay within the corpus."""
    corpus_root = corpus_dir.resolve()
    contents: dict[str, str] = {}

    def read_source(source_file: str) -> str:
        source_path = (corpus_root / source_file).resolve()
        try:
            source_path.relative_to(corpus_root)
        except ValueError as error:
            raise ValueError(f"benchmark source path escapes corpus root: {source_file}") from error
        if not source_path.is_file():
            raise ValueError(f"benchmark source does not exist: {source_file}")
        return contents.setdefault(source_file, source_path.read_text(encoding="utf-8").casefold())

    for case in cases:
        declared_sources = set(case.source_files)
        for source_file in declared_sources:
            read_source(source_file)
        for expectation in case.expected_evidence:
            if expectation.source_file not in declared_sources:
                raise ValueError(
                    f"expected evidence source is not declared by {case.id}: {expectation.source_file}"
                )
            if expectation.contains.casefold() not in read_source(expectation.source_file):
                raise ValueError(
                    f"expected phrase not found for {case.id} in {expectation.source_file}: "
                    f"{expectation.contains}"
                )
