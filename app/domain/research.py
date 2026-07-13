from typing import Literal

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    objective: str
    subquestions: list[str] = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1)
    completion_criteria: list[str] = Field(min_length=1)


class Finding(BaseModel):
    claim: str
    supporting_evidence_ids: list[str] = Field(min_length=1)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class ResearchSynthesis(BaseModel):
    findings: list[Finding]
    unresolved_questions: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    sufficient: bool
    reason: str
    evidence_gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)


class ReportFinding(BaseModel):
    heading: str
    narrative: str
    evidence_ids: list[str] = Field(min_length=1)


class DraftReport(BaseModel):
    title: str
    summary: str
    findings: list[ReportFinding]
    limitations: list[str] = Field(default_factory=list)
    markdown: str


class Citation(BaseModel):
    evidence_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    excerpt: str


class ResearchReport(BaseModel):
    title: str
    summary: str
    findings: list[ReportFinding]
    limitations: list[str]
    citations: list[Citation]
    markdown: str
    evidence_sufficient: bool
