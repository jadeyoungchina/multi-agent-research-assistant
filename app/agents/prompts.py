from app.agents.common import format_evidence
from app.domain.documents import EvidenceChunk
from app.domain.providers import ChatMessage
from app.domain.research import Critique, ResearchPlan, ResearchSynthesis


_GROUNDING_RULES = """Use only the supplied evidence.
Refer to sources exclusively by their Evidence ID.
If evidence is absent or conflicting, state the limitation explicitly.
Never create an Evidence ID or source detail."""


def planner_messages(question: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "Create a focused research plan with subquestions, search queries, "
                "and completion criteria."
            ),
        ),
        ChatMessage(role="user", content=f"Research question:\n{question}"),
    ]


def researcher_messages(
    question: str, plan: ResearchPlan, evidence: list[EvidenceChunk]
) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "Synthesize findings that answer the research question. Copy Evidence IDs exactly "
                "(for example ref_0001); omit the display label 'Evidence ID: '. "
                "If no supplied passage answers the question, findings may be empty; explain the "
                "missing evidence in unresolved_questions. Never invent placeholder IDs.\n\n"
                f"{_GROUNDING_RULES}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Research question:\n{question}\n\n"
                f"Research plan:\n{plan.model_dump_json(indent=2)}\n\n"
                f"Supplied evidence:\n{format_evidence(evidence)}"
            ),
        ),
    ]


def critic_messages(
    question: str,
    plan: ResearchPlan,
    synthesis: ResearchSynthesis,
    evidence: list[EvidenceChunk],
) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "Assess whether the synthesis is sufficiently grounded and identify "
                "gaps or follow-up queries. Check every plan completion criterion "
                "for semantic coverage, accepting faithful paraphrases. If any required "
                "criterion is unmet, set sufficient=false, name it in evidence_gaps, "
                "and suggest focused follow_up_queries.\n\n"
                f"{_GROUNDING_RULES}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Research question:\n{question}\n\n"
                f"Research plan:\n{plan.model_dump_json(indent=2)}\n\n"
                f"Synthesis:\n{synthesis.model_dump_json(indent=2)}\n\n"
                f"Supplied evidence:\n{format_evidence(evidence)}"
            ),
        ),
    ]


def writer_messages(
    question: str,
    synthesis: ResearchSynthesis,
    critique: Critique,
    evidence: list[EvidenceChunk],
) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "Return a concise JSON DraftReport, with Markdown only inside its markdown string. "
                "Keep the entire JSON reply under 1200 tokens: at most three short findings, "
                "a two-sentence summary and a compact Markdown answer. Do not repeat limitations. "
                "The markdown field is the user-visible answer: citations located only in summary "
                "or findings.narrative do not count. Put citations directly INSIDE markdown. "
                "Each finding's evidence_ids must also appear as inline citation tokens in markdown. "
                "Cite every evidence-backed statement in the summary and findings of markdown "
                "with the exact token [[cite:{Evidence ID}]]. Do not include filenames, "
                "page numbers, or invented reference labels; the deterministic validator "
                  "supplies source metadata. Describe missing document information in plain "
                  "language. Do not expose internal agent names, critique reasons, completion "
                  "criteria, scoring or revision bookkeeping in the report.\n\n"
                f"{_GROUNDING_RULES}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Research question:\n{question}\n\n"
                f"Synthesis:\n{synthesis.model_dump_json(indent=2)}\n\n"
                  f"Evidence sufficient: {critique.sufficient}\n\n"
                f"Supplied evidence:\n{format_evidence(evidence)}"
            ),
        ),
    ]
