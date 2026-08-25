from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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
