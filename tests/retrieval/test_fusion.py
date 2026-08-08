import pytest

from app.domain.documents import EvidenceChunk
from app.retrieval.fusion import normalize_queries, reciprocal_rank_fuse


def chunk(evidence_id: str, *, filename: str | None = None) -> EvidenceChunk:
    return EvidenceChunk(
        id=evidence_id,
        document_id="doc1",
        filename=filename or f"{evidence_id}.pdf",
        page_number=3,
        chunk_index=7,
        content=f"content for {evidence_id}",
        content_sha256=f"sha-{evidence_id}",
        embedding_model="test-embedding",
        score=0.9,
    )


def test_normalize_queries_keeps_original_first_and_deduplicates() -> None:
    assert normalize_queries(
        " Main question ", ["query", "QUERY", "extra"], max_expansions=2
    ) == ["Main question", "query", "extra"]


def test_normalize_queries_counts_only_unique_non_blank_expansions() -> None:
    assert normalize_queries(
        "Question",
        [" ", "QUESTION", "first", "FIRST", "second", "third"],
        max_expansions=2,
    ) == ["Question", "first", "second"]


def test_normalize_queries_with_zero_expansions_keeps_only_question() -> None:
    assert normalize_queries("Question", ["extra"], max_expansions=0) == ["Question"]


def test_normalize_queries_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="question"):
        normalize_queries("   ", ["extra"], max_expansions=1)


def test_normalize_queries_rejects_negative_expansion_limit() -> None:
    with pytest.raises(ValueError, match="max_expansions"):
        normalize_queries("Question", ["extra"], max_expansions=-1)


def test_rrf_rewards_evidence_seen_in_multiple_rankings() -> None:
    result = reciprocal_rank_fuse(
        [[chunk("a"), chunk("b")], [chunk("b"), chunk("c")]],
        rrf_k=60,
        top_k=3,
    )

    assert [item.id for item in result] == ["b", "a", "c"]


def test_rrf_returns_empty_for_empty_rankings() -> None:
    assert reciprocal_rank_fuse([], rrf_k=60, top_k=3) == []
    assert reciprocal_rank_fuse([[], []], rrf_k=60, top_k=3) == []


def test_rrf_ignores_duplicate_ids_within_one_ranking() -> None:
    result = reciprocal_rank_fuse(
        [[chunk("a"), chunk("a"), chunk("b")]], rrf_k=60, top_k=2
    )

    assert [item.id for item in result] == ["a", "b"]
    assert result[0].score == pytest.approx(1 / 61)
    assert result[1].score == pytest.approx(1 / 63)


def test_rrf_limits_results_to_top_k() -> None:
    result = reciprocal_rank_fuse(
        [[chunk("a"), chunk("b"), chunk("c")]], rrf_k=60, top_k=2
    )

    assert [item.id for item in result] == ["a", "b"]


def test_rrf_breaks_score_ties_by_evidence_id() -> None:
    result = reciprocal_rank_fuse(
        [[chunk("z")], [chunk("a")]], rrf_k=60, top_k=2
    )

    assert [item.id for item in result] == ["a", "z"]


@pytest.mark.parametrize("rrf_k", [0, -1, 1.5, True])
def test_rrf_rejects_invalid_rrf_k(rrf_k: int | float) -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fuse([], rrf_k=rrf_k, top_k=1)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_rrf_rejects_invalid_top_k(top_k: int | float) -> None:
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fuse([], rrf_k=60, top_k=top_k)


def test_rrf_preserves_evidence_metadata_and_does_not_mutate_input() -> None:
    original = chunk("a", filename="source.pdf")

    result = reciprocal_rank_fuse([[original]], rrf_k=60, top_k=1)

    assert result[0].model_dump(exclude={"score"}) == original.model_dump(
        exclude={"score"}
    )
    assert result[0].score == pytest.approx(1 / 61)
    assert original.score == 0.9
