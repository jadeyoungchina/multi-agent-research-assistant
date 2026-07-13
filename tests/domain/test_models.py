from datetime import UTC

import pytest
from pydantic import ValidationError

from app.domain.documents import EvidenceChunk
from app.domain.errors import ProviderError
from app.domain.research import Citation, Finding, ReportFinding, ResearchPlan
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


def test_domain_error_exposes_the_caller_supplied_code() -> None:
    error = ProviderError("provider_invalid_response", "invalid provider response")

    assert error.code == "provider_invalid_response"
    assert str(error) == "invalid provider response"


def test_finding_rejects_blank_supporting_evidence_id() -> None:
    with pytest.raises(ValidationError):
        Finding(claim="claim", supporting_evidence_ids=["   "], confidence="high")


def test_report_finding_rejects_blank_evidence_id() -> None:
    with pytest.raises(ValidationError):
        ReportFinding(heading="heading", narrative="narrative", evidence_ids=[""])


def test_citation_rejects_blank_evidence_id() -> None:
    with pytest.raises(ValidationError):
        Citation(
            evidence_id="",
            filename="source.md",
            page_number=None,
            chunk_index=0,
            excerpt="excerpt",
        )
