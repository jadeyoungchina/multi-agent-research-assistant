import html
import re
from pathlib import Path
from typing import Any

from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError
from app.domain.research import Citation, Critique, DraftReport, ResearchReport
from app.workflow.state import WorkflowState


_CITATION_TOKEN = re.compile(r"\[\[cite:([A-Za-z0-9_-]+)\]\]")
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]{}()#+\-!|])")
_PAGELESS_SUFFIXES = {".md", ".markdown", ".txt"}


def validate_draft_citations(
    draft: DraftReport, evidence: list[EvidenceChunk]
) -> list[str]:
    token_ids = _CITATION_TOKEN.findall(draft.markdown)
    if "[[cite:" in _CITATION_TOKEN.sub("", draft.markdown):
        raise CitationError(
            "malformed_citation_token",
            "malformed_citation_token: citation tokens must use [[cite:evidence_id]]",
        )

    valid_ids = {chunk.id for chunk in evidence}
    declared_ids = {
        evidence_id
        for finding in draft.findings
        for evidence_id in finding.evidence_ids
    }
    unknown_ids = sorted((set(token_ids) | declared_ids) - valid_ids)
    if unknown_ids:
        raise CitationError(
            "unknown_evidence_id",
            f"unknown_evidence_id: {', '.join(unknown_ids)}",
        )

    missing_ids = sorted(declared_ids - set(token_ids))
    if missing_ids:
        raise CitationError(
            "missing_citation_token",
            f"missing_citation_token: {', '.join(missing_ids)}",
        )
    return token_ids


def _limitations_for(draft: DraftReport, critique: Critique) -> list[str]:
    limitations = list(draft.limitations)
    if critique.sufficient:
        return limitations

    reason = critique.reason.strip() or "现有证据未达到充分性要求。"
    if not any(reason in limitation for limitation in limitations):
        limitations.append(f"证据不足：{reason}")
    return limitations


def _escape_markdown(value: str) -> str:
    return html.escape(_MARKDOWN_SPECIAL.sub(r"\\\1", value), quote=True)


def _render_excerpt(excerpt: str) -> str:
    lines = excerpt.splitlines() or [""]
    return "\n".join(f"> {_escape_markdown(line)}" if line else ">" for line in lines)


def _render_citation(number: int, citation: Citation) -> str:
    filename = _escape_markdown(" ".join(citation.filename.split()))
    location = filename
    if Path(citation.filename).suffix.lower() not in _PAGELESS_SUFFIXES:
        if citation.page_number is not None:
            location += f", page {citation.page_number}"
    location += f", chunk {citation.chunk_index}"
    return "\n".join(
        (
            f'<a id="citation-{citation.evidence_id}"></a>**[{number}] {location}**',
            _render_excerpt(citation.excerpt),
        )
    )


def validate_and_render_report(
    run_id: str,
    draft: DraftReport,
    evidence: list[EvidenceChunk],
    critique: Critique,
) -> ResearchReport:
    del run_id
    token_ids = validate_draft_citations(draft, evidence)
    evidence_by_id = {chunk.id: chunk for chunk in evidence}
    ordered_ids = list(dict.fromkeys(token_ids))
    numbers = {
        evidence_id: number for number, evidence_id in enumerate(ordered_ids, start=1)
    }

    rendered_draft = _CITATION_TOKEN.sub(
        lambda match: (
            f"[{numbers[match.group(1)]}](#citation-{match.group(1)})"
        ),
        draft.markdown,
    )
    citations = [
        Citation(
            evidence_id=evidence_id,
            filename=evidence_by_id[evidence_id].filename,
            page_number=evidence_by_id[evidence_id].page_number,
            chunk_index=evidence_by_id[evidence_id].chunk_index,
            excerpt=evidence_by_id[evidence_id].content[:320],
        )
        for evidence_id in ordered_ids
    ]
    limitations = _limitations_for(draft, critique)

    sections = [rendered_draft.rstrip(), "", "## 局限"]
    sections.extend(f"- {limitation}" for limitation in limitations)
    sections.extend(("", "## 引用"))
    for number, citation in enumerate(citations, start=1):
        sections.extend(("", _render_citation(number, citation)))

    return ResearchReport(
        title=draft.title,
        summary=draft.summary,
        findings=draft.findings,
        limitations=limitations,
        citations=citations,
        markdown="\n".join(sections).rstrip() + "\n",
        evidence_sufficient=critique.sufficient,
    )


class CitationValidatorNode:
    def __call__(self, state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        report = validate_and_render_report(
            state["run_id"],
            DraftReport.model_validate(state["draft"]),
            [EvidenceChunk.model_validate(item) for item in state["evidence"]],
            Critique.model_validate(state["critique"]),
        )
        return {
            "report": report.model_dump(mode="json"),
            "next_stage": "completed",
        }
