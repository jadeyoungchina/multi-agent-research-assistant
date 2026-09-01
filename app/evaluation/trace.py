"""Normalized, evaluator-owned workflow traces."""

# Adapted from trace_based_agent_evaluation.ipynb cells 5, 9, 13, 15, 19.
# Source commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46.
# Changes: gold expectations stay evaluator-only; latency is measured externally;
# retrieval, citation, answer coverage, token, loop, and error metrics are added.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import WorkflowVariant


class EvidenceSnapshot(BaseModel):
    """A workflow's normalized view of one retrieved evidence chunk."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    source_file: str
    page_number: int | None = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence id must not be blank")
        return value


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
    total_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(ge=0)
    retry_count: int = Field(default=0, ge=0)
    critic_loops: int = Field(ge=0)
    evidence_sufficient: bool | None = None
    provider: str
    model: str
    error_code: str | None = None

    @field_validator("retrieved_evidence")
    @classmethod
    def retrieved_evidence_ids_must_be_unique(
        cls, evidence: list[EvidenceSnapshot]
    ) -> list[EvidenceSnapshot]:
        ids = [snapshot.id for snapshot in evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieved evidence IDs must be unique")
        return evidence

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
