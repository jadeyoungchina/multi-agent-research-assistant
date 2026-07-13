from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class LoadedPage(BaseModel):
    page_number: int | None
    text: str


class DocumentRecord(BaseModel):
    id: str
    filename: str
    media_type: str
    sha256: str
    storage_path: str
    status: Literal["processing", "ready", "failed"]
    page_count: int = Field(ge=0)
    error_message: str | None = None
    created_at: datetime

    @classmethod
    def new(cls, filename: str, media_type: str, sha256: str) -> "DocumentRecord":
        return cls(
            id=uuid4().hex,
            filename=filename,
            media_type=media_type,
            sha256=sha256,
            storage_path="",
            status="processing",
            page_count=0,
            created_at=datetime.now(UTC),
        )


class EvidenceChunk(BaseModel):
    id: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int = Field(ge=0)
    content: str
    content_sha256: str
    embedding_model: str | None = None
    score: float | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk content must not be blank")
        return value
