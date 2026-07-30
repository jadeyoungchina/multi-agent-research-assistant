import pytest

from app.domain.documents import LoadedPage
from app.retrieval.chunking import chunk_pages, make_evidence_id


def test_chunks_do_not_cross_page_boundaries() -> None:
    """A page's chunks must never contain text from another page."""
    chunks = chunk_pages(
        document_id="doc1",
        filename="paper.pdf",
        pages=[
            LoadedPage(page_number=1, text="A" * 1500),
            LoadedPage(page_number=2, text="B" * 200),
        ],
        chunk_size=1200,
        overlap=200,
    )

    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(
        set(chunk.content) <= ({"A"} if chunk.page_number == 1 else {"B"})
        for chunk in chunks
    )


def test_chunk_ids_are_stable() -> None:
    """The same document input must produce the same traceable identifiers."""
    kwargs = dict(
        document_id="doc1",
        filename="a.md",
        pages=[LoadedPage(page_number=None, text="alpha\n\nbeta")],
        chunk_size=20,
        overlap=5,
    )

    assert [chunk.id for chunk in chunk_pages(**kwargs)] == [
        chunk.id for chunk in chunk_pages(**kwargs)
    ]


def test_changed_content_produces_changed_evidence_id() -> None:
    """Content digests must keep evidence IDs from identifying stale text."""
    original = chunk_pages(
        "doc1", "a.txt", [LoadedPage(page_number=1, text="alpha")], 100, 0
    )
    changed = chunk_pages(
        "doc1", "a.txt", [LoadedPage(page_number=1, text="bravo")], 100, 0
    )

    assert original[0].id != changed[0].id


def test_evidence_ids_distinguish_missing_and_zero_page_numbers() -> None:
    """A page number of zero must not collide with an unspecified page."""
    unspecified_page = make_evidence_id("doc1", None, 0, "digest")
    zero_page = make_evidence_id("doc1", 0, 0, "digest")

    assert unspecified_page != zero_page


@pytest.mark.parametrize("overlap", [-1, 100])
def test_overlap_must_be_non_negative_and_smaller_than_chunk_size(overlap: int) -> None:
    """Non-progressing overlap configurations must be rejected."""
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages("d", "a.txt", [LoadedPage(page_number=None, text="text")], 100, overlap)


def test_chunk_size_must_be_positive() -> None:
    """A non-positive size cannot define a valid chunk window."""
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_pages("d", "a.txt", [LoadedPage(page_number=None, text="text")], 0, 0)


def test_chunk_indexes_are_global_and_zero_based() -> None:
    """Indexes identify evidence across all pages, rather than reset per page."""
    chunks = chunk_pages(
        "doc1",
        "paper.pdf",
        [LoadedPage(page_number=1, text="first"), LoadedPage(page_number=2, text="second")],
        100,
        0,
    )

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_chunking_prefers_a_paragraph_boundary_when_available() -> None:
    """A nearby paragraph break prevents splitting the first paragraph mid-word."""
    chunks = chunk_pages(
        "doc1",
        "notes.txt",
        [LoadedPage(page_number=1, text="one two\n\nthree four")],
        10,
        0,
    )

    assert chunks[0].content == "one two"


def test_blank_page_text_does_not_produce_evidence_chunks() -> None:
    """Whitespace-only text must not produce invalid, blank evidence."""
    chunks = chunk_pages(
        "doc1",
        "empty.txt",
        [LoadedPage(page_number=1, text=" \n\t "), LoadedPage(page_number=2, text="")],
        100,
        0,
    )

    assert chunks == []
