import pytest

from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError
from app.domain.research import Critique, DraftReport, ReportFinding
from app.workflow.citations import CitationValidatorNode, validate_and_render_report


def chunk(
    evidence_id: str = "ev_good",
    *,
    filename: str = "source.pdf",
    page_number: int | None = 3,
    chunk_index: int = 2,
    content: str = "A supported excerpt.",
) -> EvidenceChunk:
    return EvidenceChunk(
        id=evidence_id,
        document_id="doc1",
        filename=filename,
        page_number=page_number,
        chunk_index=chunk_index,
        content=content,
        content_sha256="abc",
    )


def draft_with(
    evidence_id: str = "ev_good",
    *,
    markdown: str | None = None,
    evidence_ids: list[str] | None = None,
    limitations: list[str] | None = None,
) -> DraftReport:
    return DraftReport(
        title="Report",
        summary="Summary",
        findings=[
            ReportFinding(
                heading="Finding",
                narrative="Supported result",
                evidence_ids=evidence_ids or [evidence_id],
            )
        ],
        limitations=limitations or [],
        markdown=markdown or f"# Report\n\nSupported result [[cite:{evidence_id}]]",
    )


def critique(*, sufficient: bool, reason: str = "Assessment") -> Critique:
    return Critique(sufficient=sufficient, reason=reason)


def test_validator_expands_valid_ids_and_renders_markdown() -> None:
    report = validate_and_render_report(
        "run1", draft_with("ev_good"), [chunk("ev_good")], critique(sufficient=True)
    )

    assert report.citations[0].evidence_id == "ev_good"
    assert report.citations[0].filename == "source.pdf"
    assert report.citations[0].page_number == 3
    assert report.citations[0].chunk_index == 2
    assert report.citations[0].excerpt == "A supported excerpt."
    assert "[1](#citation-ev_good)" in report.markdown
    assert '<a id="citation-ev_good"></a>' in report.markdown
    assert "**[1] source.pdf, page 3, chunk 2**" in report.markdown
    assert report.evidence_sufficient is True


def test_validator_rejects_unknown_id_before_creating_report() -> None:
    with pytest.raises(CitationError, match="unknown_evidence_id"):
        validate_and_render_report(
            "run1",
            draft_with("ev_unknown"),
            [chunk("ev_good")],
            critique(sufficient=True),
        )


@pytest.mark.parametrize(
    "malformed",
    ["[[cite:]]", "[[cite:ev good]]", "[[cite:ev.good]]", "[[cite:ev_good]"],
)
def test_validator_rejects_malformed_citation_tokens(malformed: str) -> None:
    with pytest.raises(CitationError, match="malformed_citation_token"):
        validate_and_render_report(
            "run1",
            draft_with(markdown=f"Finding {malformed}"),
            [chunk()],
            critique(sufficient=True),
        )


def test_validator_rejects_uppercase_citation_fragment_beside_valid_token() -> None:
    with pytest.raises(CitationError, match="malformed_citation_token"):
        validate_and_render_report(
            "run1",
            draft_with(
                "ev",
                markdown="Valid [[cite:ev]] unsupported [[CITE:ev_unknown]]",
            ),
            [chunk("ev")],
            critique(sufficient=True),
        )


def test_validator_rejects_valid_token_with_extra_closing_bracket() -> None:
    with pytest.raises(CitationError, match="malformed_citation_token"):
        validate_and_render_report(
            "run1",
            draft_with("ev", markdown="Malformed [[cite:ev]]]"),
            [chunk("ev")],
            critique(sufficient=True),
        )


def test_validator_requires_tokens_for_all_declared_finding_evidence() -> None:
    with pytest.raises(CitationError, match="missing_citation_token"):
        validate_and_render_report(
            "run1",
            draft_with(markdown="Finding without citation"),
            [chunk()],
            critique(sufficient=True),
        )


def test_validator_orders_by_first_appearance_and_deduplicates_citations() -> None:
    report = validate_and_render_report(
        "run1",
        draft_with(
            markdown=(
                "Second [[cite:ev_b]], first [[cite:ev_a]], "
                "and second again [[cite:ev_b]]."
            ),
            evidence_ids=["ev_a", "ev_b"],
        ),
        [chunk("ev_a"), chunk("ev_b")],
        critique(sufficient=True),
    )

    assert [citation.evidence_id for citation in report.citations] == ["ev_b", "ev_a"]
    assert report.markdown.count("[1](#citation-ev_b)") == 2
    assert report.markdown.count('<a id="citation-ev_b"></a>') == 1
    assert report.markdown.count("[2](#citation-ev_a)") == 1


def test_validator_caps_excerpts_at_320_characters() -> None:
    report = validate_and_render_report(
        "run1",
        draft_with(),
        [chunk(content="x" * 400)],
        critique(sufficient=True),
    )

    assert report.citations[0].excerpt == "x" * 320
    assert len(report.citations[0].excerpt) == 320


@pytest.mark.parametrize("filename", ["notes.md", "NOTES.MARKDOWN", "notes.txt"])
def test_validator_omits_page_for_markdown_and_text_sources(filename: str) -> None:
    report = validate_and_render_report(
        "run1",
        draft_with(),
        [chunk(filename=filename, page_number=7)],
        critique(sufficient=True),
    )

    citation_line = next(
        line for line in report.markdown.splitlines() if line.startswith("<a id=")
    )
    assert "page 7" not in citation_line
    assert "chunk 2" in citation_line


def test_validator_escapes_source_filename_and_excerpt_in_markdown() -> None:
    report = validate_and_render_report(
        "run1",
        draft_with(),
        [
            chunk(
                filename="<script>_[source].pdf",
                content='<img src=x onerror="alert(1)"> *bold* [link](bad)',
            )
        ],
        critique(sufficient=True),
    )

    citation_section = report.markdown.split("## 引用", maxsplit=1)[1]
    assert "<script>" not in citation_section
    assert "<img " not in citation_section
    assert "&lt;script&gt;\\_\\[source\\].pdf" in citation_section
    assert "&lt;img src=x onerror=&quot;alert\\(1\\)&quot;&gt;" in citation_section
    assert "\\*bold\\*" in citation_section
    assert "\\[link\\]\\(bad\\)" in citation_section


def test_validator_preserves_insufficiency_after_revision_limit() -> None:
    report = validate_and_render_report(
        "run1",
        draft_with(limitations=[]),
        [chunk()],
        critique(
            sufficient=False,
            reason="The revision limit was reached before evidence became sufficient.",
        ),
    )

    assert report.evidence_sufficient is False
    assert report.limitations == [
        "证据不足：The revision limit was reached before evidence became sufficient."
    ]
    assert "## 局限" in report.markdown
    assert f"- {report.limitations[0]}" in report.markdown


def test_citation_validator_node_is_the_final_report_creator() -> None:
    state = {
        "run_id": "run1",
        "draft": draft_with().model_dump(mode="json"),
        "evidence": [chunk().model_dump(mode="json")],
        "critique": critique(sufficient=True).model_dump(mode="json"),
    }

    update = CitationValidatorNode()(state)

    assert update["next_stage"] == "completed"
    assert update["report"]["title"] == "Report"
    assert update["report"]["citations"][0]["evidence_id"] == "ev_good"
