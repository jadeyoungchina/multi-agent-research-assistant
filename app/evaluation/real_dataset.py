"""Versioned public-document suite; gold annotations never select retrieval inputs."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dataset import BenchmarkSource, preflight_source_files
from .models import EvidenceExpectation


QuestionType = Literal[
    "single_document_fact", "cross_document_synthesis", "conditions_limits_and_numbers",
    "version_and_apparent_conflict", "insufficient_corpus_evidence",
]


class HumanReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["pending", "approved", "rejected"] = "pending"
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    notes: str = ""

    @model_validator(mode="after")
    def completed_review_has_attribution(self):
        if self.status != "pending" and (not self.reviewer or self.reviewed_at is None):
            raise ValueError("completed human review requires reviewer and timestamp")
        return self


class LocatedEvidence(EvidenceExpectation):
    """One verbatim unit, with inclusive line numbers in normalized corpus text."""

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class QuestionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    method: Literal["ai_authored"] = "ai_authored"
    model: str
    task_id: str
    ai_review: Literal["pending", "checked"] = "pending"


class RealBenchmarkCase(BaseModel):
    """Evaluator-owned. Deliberately has no source_files or candidate pool field."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["real-case-v1"] = "real-case-v1"
    id: str = Field(pattern=r"^REAL-(DEV|TEST)-\d{3}$")
    split: Literal["dev", "test"]
    language: Literal["zh", "en"]
    question_type: QuestionType
    family: str
    question_group: str
    question: str = Field(min_length=10)
    answer_key_points: list[str] = Field(min_length=1)
    expected_evidence: list[LocatedEvidence]
    expected_sufficiency: bool
    abstention_policy: str = Field(min_length=10)
    insufficiency_audit: str | None = None
    provenance: QuestionProvenance
    human_review: HumanReview = Field(default_factory=HumanReview)

    @model_validator(mode="after")
    def consistent_gold(self):
        if not self.id.startswith(f"REAL-{self.split.upper()}-"):
            raise ValueError("case ID and split disagree")
        if any(not point.strip() for point in self.answer_key_points):
            raise ValueError("blank answer key point")
        insufficient = self.question_type == "insufficient_corpus_evidence"
        if insufficient != (not self.expected_sufficiency):
            raise ValueError("question type and sufficiency disagree")
        if insufficient:
            if self.expected_evidence or not self.insufficiency_audit:
                raise ValueError("insufficient cases need empty gold and a whole-pool audit rationale")
        elif not self.expected_evidence:
            raise ValueError("answerable cases need located evidence")
        return self


class CorpusLicense(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    spdx: Literal["MIT"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    family: str
    split: Literal["dev", "test"]
    repository: Literal["pydantic/pydantic", "fastapi/fastapi"]
    tag: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    upstream_path: str
    raw_path: str
    corpus_path: str
    source_url: str
    raw_url: str
    retrieved_at_utc: str
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_id: str
    original_line_count: int = Field(ge=1)
    normalized_line_offset: int = Field(ge=0)


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["real-corpus-v1"]
    dataset_kind: Literal["public_documents_ai_authored_questions"]
    normalization: str
    split_unit: Literal["topic_family_with_all_source_versions"]
    licenses: list[CorpusLicense]
    documents: list[CorpusDocument] = Field(min_length=1)


def _confined(root: Path, name: str) -> Path:
    from pathlib import PureWindowsPath
    windows = PureWindowsPath(name)
    path = (root / name).resolve()
    if windows.drive or windows.root or ".." in windows.parts or not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path must be confined")
    return path


def load_corpus_manifest(path: Path) -> CorpusManifest:
    """Offline hash/license/split/line-map validation before any provider is built."""
    manifest = CorpusManifest.model_validate_json(path.read_bytes())
    root = path.parent.resolve()
    licenses = {item.id: item for item in manifest.licenses}
    if len(licenses) != len(manifest.licenses):
        raise ValueError("duplicate license ID")
    for item in licenses.values():
        content = _confined(root, item.path).read_bytes()
        if sha256(content).hexdigest() != item.sha256 or b"Permission is hereby granted, free of charge" not in content:
            raise ValueError("license hash or MIT notice mismatch")
    groups: dict[str, str] = {}
    seen = set()
    seen_paths = set()
    for doc in manifest.documents:
        if doc.license_id not in licenses or doc.id in seen:
            raise ValueError("missing license or duplicate document ID")
        seen.add(doc.id)
        for name in (doc.raw_path, doc.corpus_path):
            if name in seen_paths:
                raise ValueError("duplicate manifest snapshot path")
            seen_paths.add(name)
        for identity in (doc.family, f"{doc.repository}/{doc.upstream_path}", doc.raw_sha256):
            if identity in groups and groups[identity] != doc.split:
                raise ValueError("document family, source version, or exact duplicate crosses splits")
            groups[identity] = doc.split
        for name, digest in ((doc.raw_path, doc.raw_sha256), (doc.corpus_path, doc.corpus_sha256)):
            if sha256(_confined(root, name).read_bytes()).hexdigest() != digest:
                raise ValueError("corpus snapshot hash mismatch")
        raw = _confined(root, doc.raw_path).read_text("utf-8").splitlines()
        normalized = _confined(root, doc.corpus_path).read_text("utf-8").splitlines()
        if len(raw) != doc.original_line_count or normalized[doc.normalized_line_offset:] != raw:
            raise ValueError("snapshot line mapping mismatch")
        expected_url = f"https://github.com/{doc.repository}/blob/{doc.commit}/{doc.upstream_path}"
        if doc.source_url != expected_url:
            raise ValueError("source URL is not pinned to the recorded commit")
    return manifest


@dataclass(frozen=True)
class RealSuite:
    manifest: CorpusManifest
    cases: list[RealBenchmarkCase]
    sources: dict[str, BenchmarkSource]

    @property
    def candidate_source_files(self) -> list[str]:
        return list(self.sources)


def load_real_suite(manifest_path: Path, dataset_path: Path, split: str, *,
                    require_human_review: bool = False, require_design_counts: bool = False) -> RealSuite:
    manifest = load_corpus_manifest(manifest_path)
    cases = [RealBenchmarkCase.model_validate_json(line) for line in dataset_path.read_text("utf-8").splitlines() if line.strip()]
    if split not in {"dev", "test"} or not cases or any(case.split != split for case in cases):
        raise ValueError("dataset must contain exactly the selected split")
    if len({case.id for case in cases}) != len(cases) or len({case.question.casefold().strip() for case in cases}) != len(cases):
        raise ValueError("duplicate case ID or question")
    selected = {doc.corpus_path.removeprefix("corpus/"): doc for doc in manifest.documents if doc.split == split}
    sources = preflight_source_files(list(selected), manifest_path.parent / "corpus", max_upload_file_bytes=None)
    for case in cases:
        if case.family not in {doc.family for doc in selected.values()}:
            raise ValueError("case family is outside its split")
        if require_human_review and case.human_review.status != "approved":
            raise ValueError("formal evaluation requires approved human references")
        for evidence in case.expected_evidence:
            doc = selected.get(evidence.source_file)
            if doc is None or doc.family != case.family:
                raise ValueError("gold evidence is outside its assigned family/split")
            lines = sources[evidence.source_file].content.decode("utf-8").splitlines()
            if evidence.end_line < evidence.start_line or evidence.end_line > len(lines):
                raise ValueError("invalid gold line range")
            if evidence.contains not in "\n".join(lines[evidence.start_line - 1:evidence.end_line]):
                raise ValueError("gold quote does not match its exact source location")
        if case.question_type == "cross_document_synthesis" and len({item.source_file for item in case.expected_evidence}) < 2:
            raise ValueError("cross-document case needs at least two source documents")
    if require_design_counts:
        expected = 2 if split == "dev" else 4
        distribution = Counter((case.question_type, case.language) for case in cases)
        types = QuestionType.__args__
        if distribution != Counter({(kind, lang): expected for kind in types for lang in ("zh", "en")}):
            raise ValueError("split does not match the prespecified type/language balance")
    return RealSuite(manifest, cases, sources)


def validate_real_benchmark(root: Path) -> tuple[RealSuite, RealSuite]:
    """Validate both held-out groups together, including shared paraphrase IDs."""
    dev, test = (load_real_suite(root / "corpus-manifest.json", root / f"{split}.jsonl", split,
                                require_design_counts=True) for split in ("dev", "test"))
    if {case.question_group for case in dev.cases} & {case.question_group for case in test.cases}:
        raise ValueError("question/paraphrase group crosses splits")
    if {case.question.casefold().strip() for case in dev.cases} & {case.question.casefold().strip() for case in test.cases}:
        raise ValueError("duplicate question crosses splits")
    return dev, test
