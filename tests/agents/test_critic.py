from app.agents.critic import CriticAgent
from app.domain.documents import EvidenceChunk
from app.domain.research import Critique, Finding, ResearchPlan, ResearchSynthesis
from app.providers.fake import FakeChatProvider
from app.workflow.state import WorkflowState


class CountingFakeChatProvider(FakeChatProvider):
    def __init__(self, structured_responses: list[Critique]) -> None:
        super().__init__(structured_responses=structured_responses)
        self.structured_calls = 0

    def generate_structured(self, messages, schema):  # type: ignore[no-untyped-def]
        self.structured_calls += 1
        return super().generate_structured(messages, schema)


def evidence() -> EvidenceChunk:
    return EvidenceChunk(
        id="ev-valid",
        document_id="doc-1",
        filename="source.md",
        page_number=None,
        chunk_index=0,
        content="Supported fact.",
        content_sha256="abc123",
    )


def plan(criteria: list[str] | None = None) -> ResearchPlan:
    return ResearchPlan(
        objective="Answer the question",
        subquestions=["What happened?"],
        search_queries=["what happened"],
        completion_criteria=criteria or ["Supported fact"],
    )


def state_with(
    synthesis: ResearchSynthesis,
    *,
    criteria: list[str] | None = None,
    iteration: int = 0,
    metrics: list[dict[str, object]] | None = None,
) -> WorkflowState:
    return WorkflowState(
        question="What happened?",
        plan=plan(criteria).model_dump(mode="json"),
        evidence=[evidence().model_dump(mode="json")],
        synthesis=synthesis.model_dump(mode="json"),
        iteration=iteration,
        provider_metrics=metrics or [],
    )


def valid_synthesis(claim: str = "Supported fact") -> ResearchSynthesis:
    return ResearchSynthesis(
        findings=[
            Finding(
                claim=claim,
                supporting_evidence_ids=["ev-valid"],
                confidence="high",
            )
        ]
    )


def test_critic_forces_revision_when_a_finding_has_no_valid_support() -> None:
    provider = FakeChatProvider(
        structured_responses=[Critique(sufficient=True, reason="looks good")]
    )
    state = state_with(
        ResearchSynthesis(
            findings=[
                Finding(
                    claim="Unsupported fact",
                    supporting_evidence_ids=["missing"],
                    confidence="high",
                )
            ]
        )
    )

    update = CriticAgent(provider)(state)
    critique = Critique.model_validate(update["critique"])

    assert critique.sufficient is False
    assert critique.evidence_gaps == ["one or more findings lack valid supporting evidence"]
    assert update["next_stage"] == "retriever"
    assert update["iteration"] == 1


def test_critic_forces_revision_when_synthesis_has_no_findings() -> None:
    provider = FakeChatProvider(
        structured_responses=[Critique(sufficient=True, reason="looks good")]
    )

    update = CriticAgent(provider)(state_with(ResearchSynthesis(findings=[])))
    critique = Critique.model_validate(update["critique"])

    assert critique.sufficient is False
    assert critique.evidence_gaps == ["synthesis contains no findings"]


def test_critic_forces_revision_when_plan_criteria_are_not_covered() -> None:
    provider = FakeChatProvider(
        structured_responses=[Critique(sufficient=True, reason="looks good")]
    )

    update = CriticAgent(provider)(
        state_with(valid_synthesis(), criteria=["A separate required conclusion"])
    )
    critique = Critique.model_validate(update["critique"])

    assert critique.sufficient is False
    assert critique.evidence_gaps == [
        "one or more plan completion criteria are not covered by findings"
    ]


def test_critic_preserves_the_llm_verdict_when_evidence_and_criteria_are_valid() -> None:
    llm_value = Critique(
        sufficient=True,
        reason="all checks passed",
        evidence_gaps=["optional follow-up"],
        follow_up_queries=["more detail"],
    )
    provider = FakeChatProvider(structured_responses=[llm_value])
    state = state_with(valid_synthesis())

    update = CriticAgent(provider)(state)

    assert Critique.model_validate(update["critique"]) == llm_value
    assert update["next_stage"] == "writer"
    assert update["iteration"] == 0
    assert state["evidence"][0]["id"] == "ev-valid"


def test_critic_calls_the_provider_once_and_appends_its_metadata_once() -> None:
    provider = CountingFakeChatProvider(
        structured_responses=[Critique(sufficient=True, reason="looks good")]
    )
    prior_metadata = {"provider": "previous", "model": "previous-model"}

    update = CriticAgent(provider)(
        state_with(valid_synthesis(), metrics=[prior_metadata])
    )

    assert provider.structured_calls == 1
    assert update["provider_metrics"] == [
        prior_metadata,
        {
            "provider": "fake",
            "model": "fake-chat",
            "latency_ms": update["provider_metrics"][1]["latency_ms"],
            "retries": 0,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        },
    ]


def test_critic_keeps_insufficient_verdict_when_revision_limit_is_reached() -> None:
    provider = FakeChatProvider(
        structured_responses=[Critique(sufficient=True, reason="looks good")]
    )

    update = CriticAgent(provider, max_iterations=2)(
        state_with(ResearchSynthesis(findings=[]), iteration=2)
    )

    assert Critique.model_validate(update["critique"]).sufficient is False
    assert update["next_stage"] == "writer"
    assert update["iteration"] == 2
