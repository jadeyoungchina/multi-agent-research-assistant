from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.domain.research import ResearchReport
from app.domain.runs import ResearchRun, RunStatus


class ApiErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ApiErrorResponse(BaseModel):
    error: ApiErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database_ready: bool
    providers: dict[str, bool]


class DocumentResponse(BaseModel):
    id: str
    filename: str
    media_type: str
    sha256: str
    status: str
    page_count: int
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    documents: list[DocumentResponse]


class ResearchCreateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    document_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("question")
    @classmethod
    def question_must_contain_at_least_three_characters(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError("question must contain at least three characters")
        return cleaned


class ResearchCreateResponse(BaseModel):
    run_id: str
    status: RunStatus


class ResearchRunResponse(BaseModel):
    run: ResearchRun
    report: ResearchReport | None
