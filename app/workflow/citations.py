import html
import re
from pathlib import Path
from typing import Any

from app.domain.documents import EvidenceChunk
from app.domain.errors import CitationError
from app.domain.research import Citation, Critique, DraftReport, ResearchReport
from app.workflow.state import WorkflowState


_CITATION_TOKEN = re.compile(
    r"(?<!\[)\[\[cite:([A-Za-z0-9_-]+)\]\](?!\])"
)
_CITATION_FRAGMENT = re.compile(r"\[\[+\s*cite\b", re.IGNORECASE)
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]{}()#+\-!|])")
_PAGELESS_SUFFIXES = {".md", ".markdown", ".txt"}


def validate_citation_tokens(markdown: str, declared_ids: set[str], valid_ids: set[str]) -> list[str]:
    token_ids = _CITATION_TOKEN.findall(markdown)
    if _CITATION_FRAGMENT.search(_CITATION_TOKEN.sub("", markdown)):
        raise CitationError("malformed_citation_token", "malformed_citation_token: citations must use exact [[cite:evidence_id]] tokens")
    if (set(token_ids) | declared_ids) - valid_ids:
        raise CitationError("unknown_evidence_id", "unknown_evidence_id: citation references unknown evidence")
    if declared_ids - set(token_ids):
        raise CitationError("missing_citation_token", "missing_citation_token: declared evidence is absent from the answer")
    return token_ids


def validate_draft_citations(draft: DraftReport, evidence: list[EvidenceChunk]) -> list[str]:
    declared = {identifier for finding in draft.findings for identifier in finding.evidence_ids}
    return validate_citation_tokens(draft.markdown, declared, {chunk.id for chunk in evidence})


def report_limitations(draft: DraftReport, critique: Critique, *, question: str = "") -> list[str]:
    limitations = list(draft.limitations)
    if critique.sufficient:
        return limitations
    notice = ("现有文档不足以支持完整结论；无法确认的部分未作推断。"
              if re.search(r"[\u4e00-\u9fff]", question)
              else "The available documents do not support a complete conclusion; unresolved details remain unconfirmed.")
    if notice not in limitations:
        limitations.append(notice)
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
    *, question: str = "",
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
    limitations = report_limitations(draft, critique, question=question)

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
            question=state.get("question", ""),
        )
        return {
            "report": report.model_dump(mode="json"),
            "next_stage": "completed",
        }
