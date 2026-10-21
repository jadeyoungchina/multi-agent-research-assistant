import pytest

from app.agents.common import EvidenceLabels
from app.agents.researcher import ResearcherAgent
from app.agents.writer import WriterAgent
from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError
from app.domain.research import Critique, DraftReport, Finding, ReportFinding, ResearchPlan, ResearchSynthesis
from app.providers.fake import FakeChatProvider
from app.workflow.citations import CitationValidatorNode, validate_draft_citations


def evidence(identifier="ev_good"):
    return EvidenceChunk(id=identifier,document_id="doc",filename="source.md",page_number=None,
                         chunk_index=0,content="A supported fact.",content_sha256="test")


def draft(identifier="ev_good", declared=None):
    return DraftReport(markdown=f"A supported fact. [[cite:{identifier}]]",title="Answer",summary="Fact",
                       findings=[ReportFinding(heading="Fact",narrative="A supported fact.",evidence_ids=declared or [identifier])])


@pytest.mark.parametrize("question",["What does the document establish?","文档能确认什么？"])
def test_insufficient_writer_to_final_renderer_never_copies_internal_reason(question):
    critique=Critique(sufficient=False,reason="INTERNAL CRITIC completion_criterion_4 unmet after revision 2")
    state={"run_id":"run","question":question,"evidence":[evidence().model_dump(mode="json")],
           "synthesis":ResearchSynthesis(findings=[]).model_dump(mode="json"),
           "critique":critique.model_dump(mode="json"),"provider_metrics":[]}
    update=WriterAgent(FakeChatProvider(structured_responses=[draft()]))(state)
    report=CitationValidatorNode()(state|update)["report"]
    assert "INTERNAL" not in str(report) and report["evidence_sufficient"] is False
    assert len(report["limitations"])==1
    assert state["critique"]["reason"]==critique.reason


def test_cite_prefix_maps_only_an_exact_supplied_label_and_does_not_create_tokens():
    item=evidence("ev_"+"a"*64)
    labels=EvidenceLabels([item])
    restored=labels.draft(draft("ref_0001",["Cite:ref_0001"]))
    assert validate_draft_citations(restored,[item])==[item.id]
    assert "Cite:" not in str(restored)
    invalid=labels.draft(draft("ref_0001",["Cite:ref_9999"]))
    with pytest.raises(CitationError): validate_draft_citations(invalid,[item])
    assert labels.text("An uncited sentence.")=="An uncited sentence."


@pytest.mark.parametrize("valid_repair",[True,False])
def test_researcher_passage_in_id_field_has_one_accounted_correction(valid_repair):
    bad=ResearchSynthesis(findings=[Finding(claim="A supported fact.",supporting_evidence_ids=["A supported fact."],confidence="high")])
    good=ResearchSynthesis(findings=[Finding(claim="A supported fact.",supporting_evidence_ids=["ev_good"],confidence="high")])
    provider=FakeChatProvider(structured_responses=[bad,good if valid_repair else bad])
    state={"question":"What is known?","evidence":[evidence().model_dump(mode="json")],
           "plan":ResearchPlan(objective="Answer",subquestions=["What?"],search_queries=["fact"],completion_criteria=["supported"]).model_dump(mode="json")}
    agent=ResearcherAgent(provider,max_citation_repairs=1)
    if valid_repair:
        update=agent(state)
        assert update["synthesis"]["findings"][0]["supporting_evidence_ids"]==["ev_good"]
        assert sum(m["attempt_count"] for m in update["provider_metrics"])==2
        assert sum(m["retries"] for m in update["provider_metrics"])==1
    else:
        with pytest.raises(CitationError) as error: agent(state)
        assert error.value.metadata.attempt_count==2
        assert error.value.metadata.retries==1
