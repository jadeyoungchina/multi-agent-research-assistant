from collections.abc import Iterable

from app.agents.prompts import critic_messages
from app.domain.documents import EvidenceChunk
from app.domain.providers import ChatProvider
from app.domain.research import Critique, ResearchPlan, ResearchSynthesis
from app.workflow.routing import route_after_critic
from app.workflow.state import WorkflowState


_NO_FINDINGS_GAP = "synthesis contains no findings"
_INVALID_SUPPORT_GAP = "one or more findings lack valid supporting evidence"
_UNCOVERED_CRITERIA_GAP = (
    "one or more plan completion criteria are not covered by findings"
)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _criteria_are_covered(
    completion_criteria: Iterable[str], synthesis: ResearchSynthesis
) -> bool:
    claims = "\n".join(_normalize(finding.claim) for finding in synthesis.findings)
    return all(_normalize(criterion) in claims for criterion in completion_criteria)


class CriticAgent:
    def __init__(self, provider: ChatProvider, max_iterations: int = 2) -> None:
        self._provider = provider
        self._max_iterations = max_iterations

    def __call__(self, state: WorkflowState) -> dict[str, object]:
        plan = ResearchPlan.model_validate(state["plan"])
        evidence = [EvidenceChunk.model_validate(item) for item in state["evidence"]]
        synthesis = ResearchSynthesis.model_validate(state["synthesis"])
        llm_critique, metadata = self._provider.generate_structured(
            critic_messages(state["question"], plan, synthesis, evidence), Critique
        )

        valid_ids = {item.id for item in evidence}
        supported = all(
            finding.supporting_evidence_ids
            and set(finding.supporting_evidence_ids) <= valid_ids
            for finding in synthesis.findings
        )
        audit_gaps: list[str] = []
        if not synthesis.findings:
            audit_gaps.append(_NO_FINDINGS_GAP)
        if not supported:
            audit_gaps.append(_INVALID_SUPPORT_GAP)
        if synthesis.findings and not _criteria_are_covered(
            plan.completion_criteria, synthesis
        ):
            audit_gaps.append(_UNCOVERED_CRITERIA_GAP)

        critique = llm_critique
        if audit_gaps:
            critique = llm_critique.model_copy(
                update={
                    "sufficient": False,
                    "evidence_gaps": [*llm_critique.evidence_gaps, *audit_gaps],
                }
            )

        next_stage, iteration = route_after_critic(
            state.get("iteration", 0), critique.sufficient, self._max_iterations
        )
        return {
            "critique": critique.model_dump(mode="json"),
            "next_stage": next_stage,
            "iteration": iteration,
            "provider_metrics": [
                *state.get("provider_metrics", []),
                metadata.model_dump(mode="json"),
            ],
        }
