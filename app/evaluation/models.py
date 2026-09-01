"""Strict, evaluator-owned schemas for the synthetic benchmark."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class WorkflowVariant(StrEnum):
    BASELINE_LLM = "baseline_llm"
    LLM_RAG = "llm_rag"
    SINGLE_AGENT_RAG = "single_agent_rag"
    MULTI_AGENT_RAG = "multi_agent_rag"


class EvidenceExpectation(BaseModel):
    source_file: str
    contains: str = Field(min_length=1)


class BenchmarkCase(BaseModel):
    id: str = Field(pattern=r"^BENCH-\d{3}$")
    question: str = Field(min_length=3)
    source_files: list[str] = Field(min_length=1)
    expected_evidence: list[EvidenceExpectation] = Field(min_length=1)
    answer_key_points: list[str] = Field(min_length=1)
    max_critic_loops: int = Field(default=2, ge=0, le=2)
    latency_budget_ms: int | None = Field(default=None, gt=0)

    @field_validator("answer_key_points")
    @classmethod
    def answer_key_points_are_not_blank(cls, points: list[str]) -> list[str]:
        if any(not point.strip() for point in points):
            raise ValueError("answer_key_points must not contain blank values")
        return points
