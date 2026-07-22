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
                "Synthesize findings that answer the research question.\n\n"
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
                "gaps or follow-up queries.\n\n"
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
                "Write the report in Markdown. Cite every evidence-backed statement "
                "with the exact token [[cite:{Evidence ID}]]. Do not include filenames, "
                "page numbers, or invented reference labels; the deterministic validator "
                "supplies source metadata.\n\n"
                f"{_GROUNDING_RULES}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Research question:\n{question}\n\n"
                f"Synthesis:\n{synthesis.model_dump_json(indent=2)}\n\n"
                f"Critique:\n{critique.model_dump_json(indent=2)}\n\n"
                f"Supplied evidence:\n{format_evidence(evidence)}"
            ),
        ),
    ]
