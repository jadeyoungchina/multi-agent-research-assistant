from app.domain.providers import ChatMessage
from app.domain.research import ResearchPlan
from app.providers.fake import DeterministicEmbeddingProvider, FakeChatProvider


def test_fake_chat_returns_queued_structured_response() -> None:
    expected = ResearchPlan(
        objective="answer",
        subquestions=["q1"],
        search_queries=["query"],
        completion_criteria=["cite evidence"],
    )
    provider = FakeChatProvider(structured_responses=[expected])
    actual, metadata = provider.generate_structured(
        [ChatMessage(role="user", content="question")], ResearchPlan
    )
    assert actual == expected
    assert metadata.provider == "fake"


def test_fake_embedding_is_stable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=16)
    first, first_metadata = provider.embed_query("同一段文本")
    second, second_metadata = provider.embed_query("同一段文本")
    assert first == second
    assert round(sum(value * value for value in first), 7) == 1.0
    assert first_metadata.model == second_metadata.model == "fake-hash-16"
