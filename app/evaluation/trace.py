"""Normalized, evaluator-owned workflow traces."""

# Adapted from trace_based_agent_evaluation.ipynb cells 5, 9, 13, 15, 19.
# Changes: gold expectations stay evaluator-only; latency is measured externally;
# retrieval, citation, answer coverage, token, loop, and error metrics are added.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import WorkflowVariant


class EvidenceSnapshot(BaseModel):
    """A workflow's normalized view of one retrieved evidence chunk."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    source_file: str
    page_number: int | None
    chunk_index: int
    text: str


class EvaluationTrace(BaseModel):
    """All workflow output needed by the evaluator, excluding gold targets."""

    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    case_id: str
    variant: WorkflowVariant
    status: Literal["success", "failed"]
    answer: str
    retrieved_evidence: list[EvidenceSnapshot] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    critic_loops: int = Field(ge=0)
    provider: str
    model: str
    error_code: str | None = None

    @classmethod
    def failed(cls, case_id: str, variant: WorkflowVariant, error_code: str) -> "EvaluationTrace":
        """Create a zero-cost normalized failure without exposing evaluator targets."""
        return cls(
            case_id=case_id,
            variant=variant,
            status="failed",
            answer="",
            latency_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            model_calls=0,
            critic_loops=0,
            provider="",
            model="",
            error_code=error_code,
        )
