import pytest

from app.domain.providers import ChatMessage
from app.domain.research import Critique, DraftReport, ResearchPlan, ResearchSynthesis
from app.evaluation.workflows import GroundedAnswer, QueryPlan


@pytest.mark.parametrize("question", ["How does solar storage work?", "请总结文档中的证据。"])
def test_demo_fake_can_plan_arbitrary_questions_without_queued_responses(question):
    from app.providers.fake import build_demo_fake_provider

    provider = build_demo_fake_provider()
    messages = [ChatMessage(role="user", content=f"Research question:\n{question}")]
    for _ in range(2):
        plan, metadata = provider.generate_structured(messages, ResearchPlan)
        assert plan.objective == question
        assert plan.search_queries == [question]
        assert metadata.provider == "fake" and metadata.model == "fake-chat"
        queries, _ = provider.generate_structured(messages, QueryPlan)
        assert queries.queries == [question]


def test_demo_fake_uses_only_prompt_evidence_and_cites_same_ids_across_schemas():
    from app.providers.fake import build_demo_fake_provider

    provider = build_demo_fake_provider()
    messages = [ChatMessage(role="user", content=(
        "Question:\nWhat was observed?\n\nSupplied evidence:\n"
        "Evidence ID: runtime-17\nSource: notes.md, page unknown, chunk 0\n"
        "Content: - The observed capacity was 17 MW\n- Storage lasted 4 hours\n\n"
        "Evidence ID: runtime-21\nSource: other.md, page unknown, chunk 0\n"
        "Content: Reliability reached 99 percent."
    ))]
    synthesis, _ = provider.generate_structured(messages, ResearchSynthesis)
    assert {evidence_id for finding in synthesis.findings for evidence_id in finding.supporting_evidence_ids} == {"runtime-17", "runtime-21"}
    assert "17 MW" in " ".join(finding.claim for finding in synthesis.findings)
    critique, _ = provider.generate_structured(messages, Critique)
    assert critique.sufficient
    draft, metadata = provider.generate_structured(messages, DraftReport)
    assert "[[cite:runtime-17]]" in draft.markdown and "[[cite:runtime-21]]" in draft.markdown
    assert "17 MW" in draft.markdown and "99 percent" in draft.markdown
    assert {item for finding in draft.findings for item in finding.evidence_ids} == {"runtime-17", "runtime-21"}
    answer, _ = provider.generate_structured(messages, GroundedAnswer)
    assert answer.cited_evidence_ids == ["runtime-17", "runtime-21"]
    assert "17 MW" in answer.answer_markdown and "99 percent" in answer.answer_markdown
    assert metadata.usage.total_tokens == 0


def test_demo_fake_reports_absence_of_evidence_without_inventing_citations():
    from app.providers.fake import build_demo_fake_provider

    provider = build_demo_fake_provider()
    messages = [ChatMessage(role="user", content="Question:\nUnknown fact\n\nSupplied evidence:\n")]
    answer, _ = provider.generate(messages)
    assert "evidence" in answer.casefold()
    critique, _ = provider.generate_structured(messages, Critique)
    assert not critique.sufficient
    draft, _ = provider.generate_structured(messages, DraftReport)
    assert draft.findings == [] and "[[cite:" not in draft.markdown
    assert draft.limitations
