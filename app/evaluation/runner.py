"""Deterministic orchestration and typed, secret-free benchmark reports."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import Settings

from .metrics import AggregateMetrics, CaseScore, aggregate_scores, score_case
from .models import BenchmarkCase, WorkflowVariant
from .real_dataset import RealBenchmarkCase
from .trace import EvaluationTrace
from .workflows import EvaluationWorkflow, _SAFE_CODES, _error_code, _safe_identifier


PROMPT_VERSION = "evaluation-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProviderConfiguration(BaseModel):
    """An explicit allowlist: never serialize Settings, API keys, or base URLs."""

    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    provider: str
    model: str
    embedding_provider: str
    embedding_model: str
    timeout_seconds: float
    max_retries: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    retrieval_candidate_k: int
    retrieval_rrf_k: int
    retrieval_min_similarity: float
    max_query_expansions: int
    max_critic_loops: int
    max_workflow_steps: int
    max_output_tokens: int = 2048

    @field_validator("provider", "model", "embedding_provider", "embedding_model")
    @classmethod
    def safe_label(cls, value: str) -> str:
        return _safe_identifier(value)

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProviderConfiguration":
        provider = settings.chat_provider
        embedding = settings.embedding_provider
        return cls(
            provider=provider,
            model="fake-chat" if provider == "fake" else getattr(settings, f"{provider}_chat_model"),
            embedding_provider=embedding,
            embedding_model="fake-hash-64" if embedding == "fake" else getattr(settings, f"{embedding}_embedding_model"),
            timeout_seconds=settings.provider_timeout_seconds,
            max_retries=settings.provider_max_retries,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            retrieval_top_k=settings.retrieval_top_k,
            retrieval_candidate_k=settings.retrieval_candidate_k,
            retrieval_rrf_k=settings.retrieval_rrf_k,
            retrieval_min_similarity=settings.retrieval_min_similarity,
            max_query_expansions=settings.max_query_expansions,
            max_critic_loops=settings.max_revision_iterations,
            max_workflow_steps=settings.max_workflow_steps,
            max_output_tokens=settings.provider_max_output_tokens,
        )


class ReportMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    run_at_utc: datetime
    git_commit: str
    git_dirty: bool
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prompt_version: str
    configuration: ProviderConfiguration
    synthetic: bool = True
    dataset_kind: str = "synthetic_documents_and_questions"
    references_human_reviewed: bool = False
    references_ai_reviewed: bool = False
    ai_reference_review_sha256: str | None = None
    metrics_version: str = "global-first-pass-v2"
    corpus_manifest_sha256: str | None = None


class ReportMetrics(AggregateMetrics):
    mean_retry_count: float = Field(ge=0.0)


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    metadata: ReportMetadata
    traces: list[EvaluationTrace]
    scores: list[CaseScore]
    aggregate: ReportMetrics
    by_variant: dict[WorkflowVariant, ReportMetrics]
    planned_trace_count: int = 0
    execution_complete: bool = True


def corpus_sha256(corpus_dir: Path) -> str:
    """Hash sorted relative POSIX paths and file bytes with unambiguous lengths."""
    root = corpus_dir.resolve()
    if not root.is_dir():
        raise ValueError("corpus directory does not exist")
    digest = sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError("corpus file escapes corpus root")
        name = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _aggregate(scores: list[CaseScore], traces: list[EvaluationTrace]) -> ReportMetrics:
    return ReportMetrics(
        **aggregate_scores(scores).model_dump(),
        mean_retry_count=sum(trace.retry_count for trace in traces) / len(traces) if traces else 0.0,
    )


def evaluate_benchmark(
    cases: Sequence[BenchmarkCase | RealBenchmarkCase], workflows: Mapping[WorkflowVariant, EvaluationWorkflow], *,
    corpus_dir: Path = PROJECT_ROOT / "benchmarks/corpus", settings: Settings | None = None,
    dataset_path: Path | None = None,
    on_trace: Callable[[EvaluationTrace], None] | None = None,
    resume_traces: Mapping[tuple[str, WorkflowVariant], EvaluationTrace] | None = None,
    candidate_source_files: Sequence[str] | None = None,
    corpus_manifest_path: Path | None = None,
    ai_reference_review_path: Path | None = None,
) -> EvaluationReport:
    """Run an identical case matrix, isolating failures and timing every adapter.

    Only ID, question, and a fresh source list cross the workflow boundary.
    Ingestion/setup is excluded from per-case latency and provider usage.
    """
    if not cases or not workflows:
        raise ValueError("at least one benchmark case and workflow are required")
    real = any(isinstance(case, RealBenchmarkCase) for case in cases)
    if ai_reference_review_path is not None:
        from .reference_review import reference_digest, validate_ai_reference_review
        audit = validate_ai_reference_review(ai_reference_review_path.parent.parent)
        accepted = {entry["id"]: entry["reference_sha256"] for entry in audit["reviews"]}
        if not real or any(accepted.get(case.id) != reference_digest(case) for case in cases):
            raise ValueError("AI review does not match these reference cases")
    if real and (not all(isinstance(case, RealBenchmarkCase) for case in cases) or not candidate_source_files):
        raise ValueError("real-document cases require an independent, whole-split candidate pool")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("benchmark case IDs must be unique")
    if any(not isinstance(variant, WorkflowVariant) for variant in workflows):
        raise ValueError("workflow keys must be WorkflowVariant values")
    configuration = ProviderConfiguration.from_settings(
        settings or Settings(_env_file=None, chat_provider="fake", embedding_provider="fake")
    )
    metadata = ReportMetadata(
        run_at_utc=datetime.now(timezone.utc),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip()),
        corpus_sha256=corpus_sha256(corpus_dir),
        dataset_sha256=sha256(dataset_path.read_bytes()).hexdigest() if dataset_path else None,
        prompt_version=PROMPT_VERSION,
        configuration=configuration,
        synthetic=not real,
        dataset_kind="public_documents_ai_authored_questions" if real else "synthetic_documents_and_questions",
        references_human_reviewed=real and all(case.human_review.status == "approved" for case in cases),
        references_ai_reviewed=ai_reference_review_path is not None,
        ai_reference_review_sha256=sha256(ai_reference_review_path.read_bytes()).hexdigest() if ai_reference_review_path else None,
        corpus_manifest_sha256=sha256(corpus_manifest_path.read_bytes()).hexdigest() if corpus_manifest_path else None,
    )
    traces: list[EvaluationTrace] = []
    scores: list[CaseScore] = []
    variants = [variant for variant in WorkflowVariant if variant in workflows]
    halted = False
    for case in sorted(cases, key=lambda item: item.id):
        for variant in variants:
            started = perf_counter()
            cached = (resume_traces or {}).get((case.id, variant))
            try:
                sources = list(candidate_source_files) if candidate_source_files is not None else list(case.source_files)
                trace = cached if cached is not None else workflows[variant].run(case.id, case.question, sources)
                if not isinstance(trace, EvaluationTrace) or trace.case_id != case.id or trace.variant != variant:
                    trace = EvaluationTrace.failed(case.id, variant, "invalid_trace")
                else:
                    trace = EvaluationTrace.model_validate(trace.model_dump()).model_copy(deep=True)
            except Exception as error:
                trace = EvaluationTrace.failed(case.id, variant, _error_code(error))
            if cached is None:
                trace.latency_ms = max(0, int((perf_counter() - started) * 1000))
            trace.provider = _safe_identifier(trace.provider) or configuration.provider
            trace.model = _safe_identifier(trace.model) or configuration.model
            if trace.status == "success":
                trace.error_code = None
            elif trace.error_code not in _SAFE_CODES | {"invalid_trace"}:
                trace.error_code = "workflow_failed"
            traces.append(trace)
            scores.append(score_case(case, trace))
            if on_trace is not None and cached is None:
                on_trace(trace)
            if trace.error_code in {"budget_exhausted", "budget_overrun", "budget_input_limit", "budget_unpriced_model"}:
                halted = True
                break
        if halted:
            break
    return EvaluationReport(
        metadata=metadata, traces=traces, scores=scores, aggregate=_aggregate(scores, traces),
        by_variant={
            variant: _aggregate(
                [score for score in scores if score.variant == variant],
                [trace for trace in traces if trace.variant == variant],
            )
            for variant in variants
        },
        planned_trace_count=len(cases) * len(workflows),
        execution_complete=not halted and len(traces) == len(cases) * len(workflows),
    )
