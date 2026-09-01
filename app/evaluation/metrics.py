"""Deterministic evaluator metrics for normalized workflow traces."""

# Adapted from trace_based_agent_evaluation.ipynb cells 5, 9, 13, 15, 19.
# Changes: project metrics use evaluator-owned evidence and answer targets,
# CJK-aware token F1, and workflow-normalized cost fields.

from collections import Counter
from math import ceil
import re

from pydantic import BaseModel, ConfigDict, Field

from .models import BenchmarkCase, EvidenceExpectation, WorkflowVariant
from .trace import EvaluationTrace, EvidenceSnapshot


_TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]|[a-z0-9]+")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?\u3002\uff01\uff1f]+")


class CaseScore(BaseModel):
    """Quality and cost metrics for one case/variant trace."""

    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    case_id: str
    variant: WorkflowVariant
    success: bool
    error_code: str | None = None
    retrieval_recall_at_5: float = Field(ge=0.0, le=1.0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    answer_key_point_f1: float = Field(ge=0.0, le=1.0)
    answer_key_point_coverage: float = Field(ge=0.0, le=1.0)
    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    critic_loops: int = Field(ge=0)


class AggregateMetrics(BaseModel):
    """Suite-level deterministic averages, counts, and latency percentiles."""

    model_config = ConfigDict(extra="forbid", strict=True)

    score_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    mean_retrieval_recall_at_5: float = Field(ge=0.0, le=1.0)
    mean_citation_precision: float = Field(ge=0.0, le=1.0)
    mean_evidence_coverage: float = Field(ge=0.0, le=1.0)
    mean_answer_key_point_f1: float = Field(ge=0.0, le=1.0)
    mean_answer_key_point_coverage: float = Field(ge=0.0, le=1.0)
    mean_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: int = Field(ge=0)
    p95_latency_ms: int = Field(ge=0)
    mean_prompt_tokens: float = Field(ge=0.0)
    mean_completion_tokens: float = Field(ge=0.0)
    mean_total_tokens: float = Field(ge=0.0)
    mean_model_calls: float = Field(ge=0.0)
    mean_critic_loops: float = Field(ge=0.0)


def score_case(case: BenchmarkCase, trace: EvaluationTrace) -> CaseScore:
    """Score one normalized trace without ever handing benchmark targets to a workflow."""
    success = trace.status == "success"
    if success:
        retrieval_recall_at_5 = _retrieval_recall_at_5(case.expected_evidence, trace.retrieved_evidence)
        citation_precision = _citation_precision(trace.cited_evidence_ids, trace.retrieved_evidence)
        evidence_coverage = _evidence_coverage(
            case.expected_evidence, trace.cited_evidence_ids, trace.retrieved_evidence
        )
        answer_key_point_f1, answer_key_point_coverage = _answer_key_point_metrics(
            case.answer_key_points, trace.answer
        )
    else:
        retrieval_recall_at_5 = 0.0
        citation_precision = 0.0
        evidence_coverage = 0.0
        answer_key_point_f1 = 0.0
        answer_key_point_coverage = 0.0

    return CaseScore(
        case_id=case.id,
        variant=trace.variant,
        success=success,
        error_code=trace.error_code,
        retrieval_recall_at_5=retrieval_recall_at_5,
        citation_precision=citation_precision,
        evidence_coverage=evidence_coverage,
        answer_key_point_f1=answer_key_point_f1,
        answer_key_point_coverage=answer_key_point_coverage,
        latency_ms=trace.latency_ms,
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
        total_tokens=trace.prompt_tokens + trace.completion_tokens,
        model_calls=trace.model_calls,
        critic_loops=trace.critic_loops,
    )


def percentile(values: list[int], probability: float) -> int:
    """Return a nearest-rank percentile using zero-based list indexing."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < probability <= 1.0:
        raise ValueError("percentile probability must be in (0, 1]")
    ordered = sorted(values)
    return ordered[ceil(probability * len(ordered)) - 1]


def aggregate_scores(scores: list[CaseScore]) -> AggregateMetrics:
    """Aggregate completed and failed traces without allowing one failure to abort a suite."""
    if not scores:
        return AggregateMetrics(
            score_count=0,
            success_count=0,
            failure_count=0,
            success_rate=0.0,
            failure_rate=0.0,
            mean_retrieval_recall_at_5=0.0,
            mean_citation_precision=0.0,
            mean_evidence_coverage=0.0,
            mean_answer_key_point_f1=0.0,
            mean_answer_key_point_coverage=0.0,
            mean_latency_ms=0.0,
            p50_latency_ms=0,
            p95_latency_ms=0,
            mean_prompt_tokens=0.0,
            mean_completion_tokens=0.0,
            mean_total_tokens=0.0,
            mean_model_calls=0.0,
            mean_critic_loops=0.0,
        )

    count = len(scores)
    success_count = sum(score.success for score in scores)
    failure_count = count - success_count
    return AggregateMetrics(
        score_count=count,
        success_count=success_count,
        failure_count=failure_count,
        success_rate=success_count / count,
        failure_rate=failure_count / count,
        mean_retrieval_recall_at_5=_mean(scores, "retrieval_recall_at_5"),
        mean_citation_precision=_mean(scores, "citation_precision"),
        mean_evidence_coverage=_mean(scores, "evidence_coverage"),
        mean_answer_key_point_f1=_mean(scores, "answer_key_point_f1"),
        mean_answer_key_point_coverage=_mean(scores, "answer_key_point_coverage"),
        mean_latency_ms=_mean(scores, "latency_ms"),
        p50_latency_ms=percentile([score.latency_ms for score in scores], 0.50),
        p95_latency_ms=percentile([score.latency_ms for score in scores], 0.95),
        mean_prompt_tokens=_mean(scores, "prompt_tokens"),
        mean_completion_tokens=_mean(scores, "completion_tokens"),
        mean_total_tokens=_mean(scores, "total_tokens"),
        mean_model_calls=_mean(scores, "model_calls"),
        mean_critic_loops=_mean(scores, "critic_loops"),
    )


def _retrieval_recall_at_5(
    expectations: list[EvidenceExpectation], evidence: list[EvidenceSnapshot]
) -> float:
    correct_source_evidence = [
        snapshot for snapshot in evidence if snapshot.source_file in {item.source_file for item in expectations}
    ][:5]
    return _coverage(expectations, correct_source_evidence)


def _citation_precision(cited_ids: list[str], evidence: list[EvidenceSnapshot]) -> float:
    unique_citations = set(cited_ids)
    if not unique_citations:
        return 0.0
    known_ids = {snapshot.id for snapshot in evidence}
    return len(unique_citations & known_ids) / len(unique_citations)


def _evidence_coverage(
    expectations: list[EvidenceExpectation], cited_ids: list[str], evidence: list[EvidenceSnapshot]
) -> float:
    cited = {snapshot.id: snapshot for snapshot in evidence}
    return _coverage(expectations, [cited[item] for item in set(cited_ids) & cited.keys()])


def _coverage(expectations: list[EvidenceExpectation], evidence: list[EvidenceSnapshot]) -> float:
    if not expectations:
        return 0.0
    covered = sum(
        any(
            snapshot.source_file == expectation.source_file
            and expectation.contains.casefold() in snapshot.text.casefold()
            for snapshot in evidence
        )
        for expectation in expectations
    )
    return covered / len(expectations)


def _answer_key_point_metrics(key_points: list[str], answer: str) -> tuple[float, float]:
    if not key_points:
        return 0.0, 0.0
    answer_sentences = [sentence for sentence in _SENTENCE_BOUNDARY_PATTERN.split(answer) if sentence.strip()]
    maxima = [
        max((_token_f1(point, sentence) for sentence in answer_sentences), default=0.0)
        for point in key_points
    ]
    return sum(maxima) / len(maxima), sum(value >= 0.60 for value in maxima) / len(maxima)


def _token_f1(expected: str, actual: str) -> float:
    expected_tokens = Counter(_TOKEN_PATTERN.findall(expected.casefold()))
    actual_tokens = Counter(_TOKEN_PATTERN.findall(actual.casefold()))
    if not expected_tokens or not actual_tokens:
        return 0.0
    overlap = sum((expected_tokens & actual_tokens).values())
    return 2 * overlap / (sum(expected_tokens.values()) + sum(actual_tokens.values()))


def _mean(scores: list[CaseScore], field: str) -> float:
    return sum(getattr(score, field) for score in scores) / len(scores)
