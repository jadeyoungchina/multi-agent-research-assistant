import pytest

from app.agents.common import format_evidence
from app.agents.prompts import critic_messages, researcher_messages, writer_messages
from app.domain.documents import EvidenceChunk
from app.domain.research import Critique, Finding, ResearchPlan, ResearchSynthesis


@pytest.fixture
def chunk() -> EvidenceChunk:
    return EvidenceChunk(
        id="ev-1",
        document_id="doc-1",
        filename="paper.pdf",
        page_number=2,
        chunk_index=4,
        content="The result changed in the revised experiment.",
        content_sha256="abc123",
    )


@pytest.fixture
def plan() -> ResearchPlan:
    return ResearchPlan(
        objective="Determine what changed",
        subquestions=["What did the experiment report?"],
        search_queries=["revised experiment result"],
        completion_criteria=["Ground the response in evidence"],
    )


@pytest.fixture
def synthesis() -> ResearchSynthesis:
    return ResearchSynthesis(
        findings=[
            Finding(
                claim="The result changed.",
                supporting_evidence_ids=["ev-1"],
                confidence="high",
            )
        ]
    )


def test_format_evidence_includes_stable_ids_and_sources(chunk: EvidenceChunk) -> None:
    rendered = format_evidence([chunk])

    assert f"Evidence ID: {chunk.id}" in rendered
    assert "Source: paper.pdf, page 2, chunk 4" in rendered
    assert chunk.content in rendered


def test_grounded_agent_prompts_include_evidence_and_required_rules(
    chunk: EvidenceChunk, plan: ResearchPlan, synthesis: ResearchSynthesis
) -> None:
    messages_by_agent = [
        researcher_messages("What changed?", plan, [chunk]),
        critic_messages("What changed?", plan, synthesis, [chunk]),
        writer_messages(
            "What changed?",
            synthesis,
            Critique(sufficient=True, reason="Grounded"),
            [chunk],
        ),
    ]

    for messages in messages_by_agent:
        system_prompt = messages[0].content
        user_prompt = messages[1].content
        assert "Use only the supplied evidence." in system_prompt
        assert "Refer to sources exclusively by their Evidence ID." in system_prompt
        assert "If evidence is absent or conflicting, state the limitation explicitly." in system_prompt
        assert "Never create an Evidence ID or source detail." in system_prompt
        assert "Evidence ID: ev-1" in user_prompt


def test_writer_prompt_requires_markdown_citation_tokens_and_forbids_source_metadata(
    chunk: EvidenceChunk, synthesis: ResearchSynthesis
) -> None:
    messages = writer_messages(
        "What changed?",
        synthesis,
        Critique(sufficient=True, reason="Grounded"),
        [chunk],
    )

    system_prompt = messages[0].content
    assert "Markdown" in system_prompt
    assert "[[cite:{Evidence ID}]]" in system_prompt
    assert "filenames" in system_prompt
    assert "page numbers" in system_prompt
    assert "invented reference labels" in system_prompt
