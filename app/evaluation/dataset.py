"""Load and validate the local, synthetic benchmark dataset."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath

from pydantic import ValidationError

from app.domain.errors import DocumentError
from app.retrieval.loaders import SUPPORTED, load_document

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


@dataclass(frozen=True)
class BenchmarkSource:
    media_type: str
    content: bytes
    text: str


def preflight_benchmark_sources(
    cases: Sequence[BenchmarkCase], corpus_dir: Path, *, max_upload_file_bytes: int | None,
) -> dict[str, BenchmarkSource]:
    """Validate source-only inputs before any provider setup or ingestion.

    Content hashes follow DocumentService deduplication: distinct filenames for
    identical bytes would share a document ID and cannot represent distinct sources.
    No expected evidence or answer targets are inspected here.
    """
    root = corpus_dir.resolve()
    sources: dict[str, BenchmarkSource] = {}
    filenames_by_digest: dict[str, str] = {}
    for case in cases:
        for filename in case.source_files:
            if filename in sources:
                continue
            relative = Path(filename)
            windows_path = PureWindowsPath(filename)
            if ".." in relative.parts or ".." in windows_path.parts:
                raise ValueError("benchmark source path contains parent component")
            if relative.is_absolute() or windows_path.drive or windows_path.root:
                raise ValueError("benchmark source must be a confined relative path")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("benchmark source path escapes corpus root")
            if not path.is_file():
                raise ValueError("benchmark source does not exist")
            supported = SUPPORTED.get(path.suffix.casefold())
            if supported is None:
                raise ValueError("unsupported benchmark document type")
            with path.open("rb") as source:
                content = source.read(max_upload_file_bytes + 1 if max_upload_file_bytes is not None else -1)
            if max_upload_file_bytes is not None and len(content) > max_upload_file_bytes:
                raise ValueError("benchmark source exceeds configured size limit")
            digest = sha256(content).hexdigest()
            if digest in filenames_by_digest:
                raise ValueError("ambiguous benchmark source filenames share identical content")
            filenames_by_digest[digest] = filename
            try:
                pages = load_document(content, filename, supported[0])
            except DocumentError as error:
                raise ValueError("benchmark source could not be parsed") from error
            sources[filename] = BenchmarkSource(supported[0], content, "\n".join(page.text for page in pages))
    return sources


def validate_benchmark_corpus(
    cases: list[BenchmarkCase], corpus_dir: Path, *, max_upload_file_bytes: int | None = None,
) -> None:
    """Preflight all sources, then validate evaluator-owned evidence expectations."""
    sources = preflight_benchmark_sources(cases, corpus_dir, max_upload_file_bytes=max_upload_file_bytes)

    for case in cases:
        declared_sources = set(case.source_files)
        for expectation in case.expected_evidence:
            if expectation.source_file not in declared_sources:
                raise ValueError(
                    f"expected evidence source is not declared by {case.id}: {expectation.source_file}"
                )
            if expectation.contains.casefold() not in sources[expectation.source_file].text.casefold():
                raise ValueError(
                    f"expected phrase not found for {case.id} in {expectation.source_file}: "
                    f"{expectation.contains}"
                )
