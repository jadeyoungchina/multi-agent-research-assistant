from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


RunStatus = Literal["queued", "running", "completed", "failed"]


class ResearchRun(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    question: str
    document_ids: list[str]
    status: RunStatus
    current_stage: str
    last_completed_stage: str | None = None
    iteration: int
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_call_count: int = 0
    retry_count: int = 0
    evidence_sufficient: bool | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def new(
        cls, question: str, document_ids: list[str], provider: str, model: str
    ) -> "ResearchRun":
        now = datetime.now(UTC)
        return cls(
            id=uuid4().hex,
            question=question,
            document_ids=document_ids,
            status="queued",
            current_stage="planner",
            iteration=0,
            provider=provider,
            model=model,
            created_at=now,
            updated_at=now,
        )


class RunEvent(BaseModel):
    sequence: int = Field(ge=1)
    run_id: str
    stage: str
    event_type: Literal["created", "started", "completed", "retry", "failed", "finished"]
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
