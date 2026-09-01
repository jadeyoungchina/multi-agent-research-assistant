"""Release thresholds: only multi_agent_rag blocks the v1.0 release."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .models import WorkflowVariant
from .runner import EvaluationReport, PROJECT_ROOT


class QualityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    minimum_success_rate: float = Field(ge=0.0, le=1.0)
    minimum_retrieval_recall_at_5: float = Field(ge=0.0, le=1.0)
    minimum_citation_precision: float = Field(ge=0.0, le=1.0)
    minimum_evidence_coverage: float = Field(ge=0.0, le=1.0)
    minimum_answer_key_point_coverage: float = Field(ge=0.0, le=1.0)
    maximum_failure_rate: float = Field(ge=0.0, le=1.0)


class QualityGateError(Exception):
    def __init__(self, failures: list[str]) -> None:
        self.failures = failures
        super().__init__("Quality gate FAIL:\n" + "\n".join(failures))


def enforce_quality_gate(
    report: EvaluationReport, gate_path: Path = PROJECT_ROOT / "benchmarks/quality-gates.json",
) -> None:
    """Raise one error listing every threshold violation, never just the first."""
    gates = TypeAdapter(dict[WorkflowVariant, QualityThresholds]).validate_json(gate_path.read_text("utf-8"))
    variant = WorkflowVariant.MULTI_AGENT_RAG
    if variant not in gates:
        raise ValueError("quality gate configuration requires multi_agent_rag thresholds")
    metrics = report.by_variant.get(variant)
    if metrics is None or metrics.score_count == 0:
        raise QualityGateError(["multi_agent_rag was not evaluated"])
    fields = {
        "minimum_success_rate": "success_rate",
        "minimum_retrieval_recall_at_5": "mean_retrieval_recall_at_5",
        "minimum_citation_precision": "mean_citation_precision",
        "minimum_evidence_coverage": "mean_evidence_coverage",
        "minimum_answer_key_point_coverage": "mean_answer_key_point_coverage",
        "maximum_failure_rate": "failure_rate",
    }
    failures = []
    for threshold, field in fields.items():
        actual = getattr(metrics, field)
        bound = getattr(gates[variant], threshold)
        minimum = threshold.startswith("minimum_")
        if (minimum and actual < bound) or (not minimum and actual > bound):
            failures.append(f"{variant.value}.{threshold}: actual {actual:.6g}, required {'>=' if minimum else '<='} {bound:.6g}")
    if failures:
        raise QualityGateError(failures)
