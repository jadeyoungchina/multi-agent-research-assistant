import pytest

from app.agents.researcher import ResearcherAgent
from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError
from app.domain.providers import ProviderMetadata
from app.domain.research import Finding, ResearchPlan, ResearchSynthesis
from app.providers.fake import FakeChatProvider
from app.workflow.state import initial_state


def state_with_evidence(*evidence_ids: str) -> dict:
    state = initial_state("run", "question", ["doc"])
    state["plan"] = ResearchPlan(
        objective="compare",
        subquestions=["what agrees?"],
        search_queries=["agreement"],
        completion_criteria=["cite every finding"],
    ).model_dump(mode="json")
    state["evidence"] = [
        EvidenceChunk(
            id=evidence_id,
            document_id="doc",
            filename="source.txt",
            page_number=None,
            chunk_index=index,
            content=f"content for {evidence_id}",
            content_sha256=f"hash-{evidence_id}",
            score=0.5,
        ).model_dump(mode="json")
        for index, evidence_id in enumerate(evidence_ids)
    ]
    return state


@pytest.mark.parametrize(
    ("supporting_ids", "conflicting_ids"),
    [(["unknown"], []), (["ev-valid"], ["unknown"])],
)
def test_researcher_rejects_every_unknown_evidence_id(
    supporting_ids: list[str], conflicting_ids: list[str]
) -> None:
    """Removing either ID collection branch must cause this test to fail."""
    synthesis = ResearchSynthesis(
        findings=[
            Finding(
                claim="claim",
                supporting_evidence_ids=supporting_ids,
                conflicting_evidence_ids=conflicting_ids,
                confidence="high",
            )
        ]
    )
    provider = FakeChatProvider(structured_responses=[synthesis])

    with pytest.raises(CitationError) as error:
        ResearcherAgent(provider)(state_with_evidence("ev-valid"))

    assert error.value.code == "unknown_evidence_id"
    assert error.value.metadata.attempt_count == 1


def test_researcher_stores_grounded_synthesis_with_supporting_and_conflicting_ids() -> None:
    synthesis = ResearchSynthesis(
        findings=[
            Finding(
                claim="the sources disagree",
                supporting_evidence_ids=["ev-support"],
                conflicting_evidence_ids=["ev-conflict"],
                confidence="medium",
            )
        ]
    )
    provider = FakeChatProvider(structured_responses=[synthesis])
    state = state_with_evidence("ev-support", "ev-conflict")
    state["provider_metrics"] = [
        ProviderMetadata(provider="retrieval", model="embedding", latency_ms=1).model_dump(
            mode="json"
        )
    ]

    update = ResearcherAgent(provider)(state)

    assert update["synthesis"] == synthesis.model_dump(mode="json")
    assert update["next_stage"] == "critic"
    assert [metric["provider"] for metric in update["provider_metrics"]] == [
        "retrieval",
        "fake",
    ]


@pytest.mark.parametrize("label", ["Evidence ID: ev_", ""])
def test_format_only_id_normalization_requires_an_exact_allowed_hash(label):
    digest = "a" * 64
    synthesis = ResearchSynthesis(findings=[Finding(claim="supported", supporting_evidence_ids=[label + digest], confidence="high")])
    provider = FakeChatProvider(structured_responses=[synthesis])
    update = ResearcherAgent(provider)(state_with_evidence("ev_" + digest))
    assert update["synthesis"]["findings"][0]["supporting_evidence_ids"] == ["ev_" + digest]
    assert synthesis.findings[0].supporting_evidence_ids == [label + digest]
    invalid = synthesis.model_copy(deep=True)
    invalid.findings[0].supporting_evidence_ids = [label + "a" * 63 + "b"]
    with pytest.raises(CitationError):
        ResearcherAgent(FakeChatProvider(structured_responses=[invalid]))(state_with_evidence("ev_" + digest))
