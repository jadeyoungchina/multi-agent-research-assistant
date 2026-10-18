import pytest

from app.agents.writer import WriterAgent
from app.domain.errors import CitationError
from app.domain.research import Critique, DraftReport, ReportFinding, ResearchSynthesis
from app.providers.fake import FakeChatProvider


def state_with_evidence(evidence_id: str = "ev_good") -> dict[str, object]:
    return {
        "question": "What does the evidence show?",
        "evidence": [
            {
                "id": evidence_id,
                "document_id": "doc1",
                "filename": "source.md",
                "page_number": None,
                "chunk_index": 0,
                "content": "Supported evidence.",
                "content_sha256": "abc",
            }
        ],
        "synthesis": ResearchSynthesis(findings=[]).model_dump(mode="json"),
        "critique": Critique(sufficient=True, reason="Enough evidence").model_dump(
            mode="json"
        ),
        "provider_metrics": [],
    }


def draft_with(
    evidence_ids: list[str], markdown: str, limitations: list[str] | None = None
) -> DraftReport:
    return DraftReport(
        title="Report",
        summary="Summary",
        findings=[
            ReportFinding(
                heading="Finding",
                narrative="Supported result",
                evidence_ids=evidence_ids,
            )
        ],
        limitations=limitations or [],
        markdown=markdown,
    )


def test_writer_preserves_only_finding_evidence_ids() -> None:
    draft = draft_with(
        ["ev_good"], "# Report\n\n## Finding\nSupported result [[cite:ev_good]]"
    )
    provider = FakeChatProvider(structured_responses=[draft])

    update = WriterAgent(provider)(state_with_evidence("ev_good"))

    assert update["draft"]["findings"][0]["evidence_ids"] == ["ev_good"]
    assert update["next_stage"] == "citation_validator"
    assert update["provider_metrics"][0]["provider"] == "fake"


@pytest.mark.parametrize(
    ("evidence_ids", "markdown"),
    [
        (["ev_unknown"], "Finding [[cite:ev_unknown]]"),
        (["ev_good"], "Finding [[cite:ev_unknown]] [[cite:ev_good]]"),
    ],
)
def test_writer_rejects_unknown_declared_or_inline_evidence_ids(
    evidence_ids: list[str], markdown: str
) -> None:
    provider = FakeChatProvider(
        structured_responses=[draft_with(evidence_ids, markdown)]
    )

    with pytest.raises(CitationError, match="unknown_evidence_id"):
        WriterAgent(provider)(state_with_evidence())


def test_writer_requires_every_declared_finding_id_in_markdown() -> None:
    provider = FakeChatProvider(
        structured_responses=[draft_with(["ev_good"], "Finding without a citation")]
    )

    with pytest.raises(CitationError, match="missing_citation_token") as caught:
        WriterAgent(provider)(state_with_evidence())
    assert caught.value.metadata.provider == "fake"
    assert caught.value.metadata.attempt_count == 1


def test_writer_adds_limitation_when_critique_remains_insufficient() -> None:
    state = state_with_evidence()
    state["critique"] = Critique(
        sufficient=False,
        reason="The revision limit was reached with conflicting evidence.",
        evidence_gaps=["conflicting evidence"],
    ).model_dump(mode="json")
    provider = FakeChatProvider(
        structured_responses=[
            draft_with(["ev_good"], "Finding [[cite:ev_good]]")
        ]
    )

    update = WriterAgent(provider)(state)

    assert update["draft"]["limitations"] == [
        "证据不足：The revision limit was reached with conflicting evidence."
    ]


def test_writer_does_not_duplicate_existing_insufficiency_limitation() -> None:
    state = state_with_evidence()
    state["critique"] = Critique(
        sufficient=False,
        reason="Evidence remains incomplete.",
    ).model_dump(mode="json")
    provider = FakeChatProvider(
        structured_responses=[
            draft_with(
                ["ev_good"],
                "Finding [[cite:ev_good]]",
                limitations=["Evidence remains incomplete."],
            )
        ]
    )

    update = WriterAgent(provider)(state)

    assert update["draft"]["limitations"] == ["Evidence remains incomplete."]


def test_bounded_citation_repair_uses_feedback_and_accounts_both_responses():
    class Recording(FakeChatProvider):
        def generate_structured(self, messages, schema):
            self.messages = messages
            return super().generate_structured(messages, schema)
    provider = Recording(structured_responses=[
        draft_with(["ev_good"], "Uncited answer"),
        draft_with(["ev_good"], "Supported result [[cite:ev_good]]"),
    ])
    update = WriterAgent(provider, max_citation_repairs=1)(state_with_evidence())
    assert update["draft"]["markdown"] == "Supported result [[cite:ev_good]]"
    assert len(update["provider_metrics"]) == 2
    assert sum(item["attempt_count"] for item in update["provider_metrics"]) == 2
    assert sum(item["retries"] for item in update["provider_metrics"]) == 1
    assert "missing_citation_token" in provider.messages[-1].content
    assert all("gold" not in message.content.casefold() for message in provider.messages)


def test_failed_citation_repair_stops_and_preserves_all_usage():
    provider = FakeChatProvider(structured_responses=[
        draft_with(["ev_good"], "Uncited answer"), draft_with(["ev_good"], "Still uncited"),
    ])
    with pytest.raises(CitationError) as caught:
        WriterAgent(provider, max_citation_repairs=1)(state_with_evidence())
    assert caught.value.metadata.attempt_count == 2
    assert caught.value.metadata.retries == 1


def test_provider_failure_during_repair_keeps_the_previous_paid_response():
    from app.domain.errors import ProviderError
    from app.domain.providers import ProviderMetadata, TokenUsage
    class Failing(FakeChatProvider):
        calls = 0
        def generate_structured(self, messages, schema):
            self.calls += 1
            if self.calls == 2:
                raise ProviderError("provider_timeout", "safe", metadata=ProviderMetadata(
                    provider="fake", model="fake-chat", latency_ms=1, usage=TokenUsage(prompt_tokens=7),
                    attempt_count=1, attempt_ids=["failed-attempt"], usage_complete=False))
            value, meta = super().generate_structured(messages, schema)
            meta.usage.prompt_tokens = 5
            meta.attempt_ids = ["first-attempt"]
            return value, meta
    provider = Failing(structured_responses=[draft_with(["ev_good"], "Uncited answer")])
    with pytest.raises(ProviderError) as caught:
        WriterAgent(provider, max_citation_repairs=1)(state_with_evidence())
    assert caught.value.metadata.usage.prompt_tokens == 12
    assert caught.value.metadata.attempt_count == 2
    assert caught.value.metadata.attempt_ids == ["first-attempt", "failed-attempt"]
    assert not caught.value.metadata.usage_complete
    assert caught.value.metadata.retries == 1
