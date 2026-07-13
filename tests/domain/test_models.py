from datetime import UTC

import pytest
from pydantic import ValidationError

from app.domain.documents import EvidenceChunk
from app.domain.research import Finding, ResearchPlan
from app.domain.runs import ResearchRun


def test_plan_requires_at_least_one_search_query() -> None:
    with pytest.raises(ValidationError):
        ResearchPlan(
            objective="compare evidence",
            subquestions=["what changed?"],
            search_queries=[],
            completion_criteria=["cite a source"],
        )


def test_finding_requires_supporting_evidence() -> None:
    with pytest.raises(ValidationError):
        Finding(claim="claim", supporting_evidence_ids=[], confidence="high")


def test_chunk_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        EvidenceChunk(
            id="c1",
            document_id="d1",
            filename="a.md",
            page_number=None,
            chunk_index=0,
            content="   ",
            content_sha256="abc",
        )


def test_run_is_timezone_aware() -> None:
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    assert run.created_at.tzinfo is UTC
    assert run.status == "queued"
