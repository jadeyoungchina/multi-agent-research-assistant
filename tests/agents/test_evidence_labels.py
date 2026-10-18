import pytest

from app.agents.common import EvidenceLabels
from app.domain.documents import EvidenceChunk
from app.domain.research import DraftReport, ReportFinding
from app.workflow.citations import validate_draft_citations
from app.domain.errors import CitationError


def evidence(identifier):
    return EvidenceChunk(id=identifier, document_id="doc", filename="source.md", page_number=None,
                         chunk_index=0, content="Original source text.", content_sha256="hash")


def test_short_labels_round_trip_without_altering_sources_or_adding_citations():
    identifier = "ev_" + "a" * 64
    sources = [evidence(identifier)]
    labels = EvidenceLabels(sources)
    assert labels.evidence[0].id == "ref_0001"
    assert labels.evidence[0].content == sources[0].content
    assert sources[0].id == identifier
    draft = DraftReport(title="Report", summary="Summary", findings=[ReportFinding(
        heading="Fact", narrative="The source text.", evidence_ids=["ref_0001"])],
        markdown="The source text. [[cite:ref_0001]]")
    restored = labels.draft(draft)
    assert restored.findings[0].evidence_ids == [identifier]
    assert restored.markdown == f"The source text. [[cite:{identifier}]]"
    assert labels.draft(restored, to_provider=True) == draft
    validate_draft_citations(restored, sources)
    draft.markdown = "Uncited text."
    with pytest.raises(CitationError):
        validate_draft_citations(labels.draft(draft), sources)
    draft.markdown = "Unknown reference. [[cite:ref_9999]]"
    with pytest.raises(CitationError):
        validate_draft_citations(labels.draft(draft), sources)


def test_aliases_do_not_collide_with_existing_ids_and_never_fuzzy_match():
    identifier = "ev_" + "b" * 64
    labels = EvidenceLabels([evidence(identifier), evidence("ref_0001")])
    assert labels.forward[identifier] == "ref_0001_"
    assert labels.original("ref_0001") == "ref_0001"
    assert labels.original("ref_0001_") == identifier
    changed = "ev_" + "b" * 63 + "c"
    assert labels.original(changed) == changed
