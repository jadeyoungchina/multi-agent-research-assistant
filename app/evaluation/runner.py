"""Deterministic orchestration and typed, secret-free benchmark reports."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import subprocess
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import Settings

from .metrics import AggregateMetrics, CaseScore, aggregate_scores, score_case
from .models import BenchmarkCase, WorkflowVariant
from .trace import EvaluationTrace
from .workflows import EvaluationWorkflow, _SAFE_CODES, _error_code, _safe_identifier


PROMPT_VERSION = "evaluation-v1"
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


class ReportMetrics(AggregateMetrics):
    mean_retry_count: float = Field(ge=0.0)


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    metadata: ReportMetadata
    traces: list[EvaluationTrace]
    scores: list[CaseScore]
    aggregate: ReportMetrics
    by_variant: dict[WorkflowVariant, ReportMetrics]


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
    cases: Sequence[BenchmarkCase], workflows: Mapping[WorkflowVariant, EvaluationWorkflow], *,
    corpus_dir: Path = PROJECT_ROOT / "benchmarks/corpus", settings: Settings | None = None,
    dataset_path: Path | None = None,
) -> EvaluationReport:
    """Run an identical case matrix, isolating failures and timing every adapter.

    Only ID, question, and a fresh source list cross the workflow boundary.
    Ingestion/setup is excluded from per-case latency and provider usage.
    """
    if not cases or not workflows:
        raise ValueError("at least one benchmark case and workflow are required")
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
    )
    traces: list[EvaluationTrace] = []
    scores: list[CaseScore] = []
    variants = [variant for variant in WorkflowVariant if variant in workflows]
    for case in sorted(cases, key=lambda item: item.id):
        for variant in variants:
            started = perf_counter()
            try:
                trace = workflows[variant].run(case.id, case.question, list(case.source_files))
                if not isinstance(trace, EvaluationTrace) or trace.case_id != case.id or trace.variant != variant:
                    trace = EvaluationTrace.failed(case.id, variant, "invalid_trace")
                else:
                    trace = EvaluationTrace.model_validate(trace.model_dump()).model_copy(deep=True)
            except Exception as error:
                trace = EvaluationTrace.failed(case.id, variant, _error_code(error))
            trace.latency_ms = max(0, int((perf_counter() - started) * 1000))
            trace.provider = _safe_identifier(trace.provider) or configuration.provider
            trace.model = _safe_identifier(trace.model) or configuration.model
            if trace.status == "success":
                trace.error_code = None
            elif trace.error_code not in _SAFE_CODES | {"invalid_trace"}:
                trace.error_code = "workflow_failed"
            traces.append(trace)
            scores.append(score_case(case, trace))
    return EvaluationReport(
        metadata=metadata, traces=traces, scores=scores, aggregate=_aggregate(scores, traces),
        by_variant={
            variant: _aggregate(
                [score for score in scores if score.variant == variant],
                [trace for trace in traces if trace.variant == variant],
            )
            for variant in variants
        },
    )
