from datetime import UTC, datetime

import pytest

from app.domain.documents import DocumentRecord, EvidenceChunk
from app.domain.errors import RetrievalError
from app.retrieval.index import LocalVectorIndex
from app.storage.database import Database
from app.storage.documents import DocumentRepository


@pytest.fixture
def repository(tmp_path) -> DocumentRepository:
    database = Database(tmp_path / "index.db")
    database.initialize()
    return DocumentRepository(database)


def seed_vectors(
    repository: DocumentRepository,
    vectors: dict[str, list[float]],
    document_ids: dict[str, str] | None = None,
) -> None:
    document_ids = document_ids or {}
    documents: dict[str, DocumentRecord] = {}
    chunks: dict[str, list[tuple[EvidenceChunk, list[float]]]] = {}
    for index, (chunk_id, vector) in enumerate(vectors.items()):
        document_id = document_ids.get(chunk_id, "doc1")
        if document_id not in documents:
            document = DocumentRecord(
                id=document_id,
                filename=f"{document_id}.txt",
                media_type="text/plain",
                sha256=f"sha-{document_id}",
                storage_path=f"uploads/{document_id}.txt",
                status="ready",
                page_count=1,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            documents[document_id] = document
            chunks[document_id] = []
        document = documents[document_id]
        chunks[document_id].append(
            (
                EvidenceChunk(
                    id=chunk_id,
                    document_id=document_id,
                    filename=document.filename,
                    page_number=1,
                    chunk_index=index,
                    content=f"content for {chunk_id}",
                    content_sha256=f"content-sha-{chunk_id}",
                    embedding_model="test-model",
                ),
                vector,
            )
        )
    for document_id, document in documents.items():
        repository.add_document(document)
        repository.replace_chunks(document_id, chunks[document_id])


def test_search_orders_by_cosine_similarity(repository: DocumentRepository) -> None:
    seed_vectors(
        repository,
        {"a": [1.0, 0.0], "b": [0.8, 0.2], "c": [0.0, 1.0]},
    )

    results = LocalVectorIndex(repository).search(
        [1.0, 0.0], ["doc1"], limit=2, min_score=0.15
    )

    assert [item.id for item in results] == ["a", "b"]
    assert results[0].score == pytest.approx(1.0)


def test_search_filters_documents_and_minimum_score(
    repository: DocumentRepository,
) -> None:
    seed_vectors(
        repository,
        {"a": [1.0, 0.0], "c": [0.0, 1.0]},
        document_ids={"a": "doc1", "c": "doc2"},
    )

    results = LocalVectorIndex(repository).search(
        [1.0, 0.0], ["doc1"], limit=10, min_score=0.2
    )

    assert [item.id for item in results] == ["a"]


@pytest.mark.parametrize("query", [[0.0, 0.0], [float("nan"), 0.0], [float("inf"), 0.0]])
def test_search_rejects_invalid_query_vectors(
    repository: DocumentRepository, query: list[float]
) -> None:
    seed_vectors(repository, {"a": [1.0, 0.0]})

    with pytest.raises(RetrievalError, match="query embedding") as error:
        LocalVectorIndex(repository).search(query, ["doc1"], limit=1, min_score=0.0)

    assert error.value.code == "invalid_query_embedding"


@pytest.mark.parametrize("candidate", [[0.0, 0.0], [float("nan"), 0.0], [float("inf"), 0.0]])
def test_search_rejects_invalid_stored_vectors(
    repository: DocumentRepository, candidate: list[float]
) -> None:
    seed_vectors(repository, {"a": candidate})

    with pytest.raises(RetrievalError, match="stored embedding a") as error:
        LocalVectorIndex(repository).search(
            [1.0, 0.0], ["doc1"], limit=1, min_score=0.0
        )

    assert error.value.code == "invalid_stored_embedding"


def test_search_rejects_dimension_mismatch(repository: DocumentRepository) -> None:
    seed_vectors(repository, {"a": [1.0, 0.0, 0.0]})

    with pytest.raises(RetrievalError, match="dimensions differ") as error:
        LocalVectorIndex(repository).search(
            [1.0, 0.0], ["doc1"], limit=1, min_score=0.0
        )

    assert error.value.code == "embedding_dimension_mismatch"


def test_search_breaks_equal_scores_by_evidence_id(
    repository: DocumentRepository,
) -> None:
    seed_vectors(repository, {"z": [1.0, 0.0], "a": [1.0, 0.0]})

    results = LocalVectorIndex(repository).search(
        [1.0, 0.0], ["doc1"], limit=2, min_score=0.0
    )

    assert [item.id for item in results] == ["a", "z"]


def test_search_returns_empty_for_an_empty_document_set(
    repository: DocumentRepository,
) -> None:
    seed_vectors(repository, {"a": [1.0, 0.0]})

    assert LocalVectorIndex(repository).search(
        [1.0, 0.0], [], limit=1, min_score=0.0
    ) == []


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_search_rejects_invalid_result_limits(
    repository: DocumentRepository, limit: int | float
) -> None:
    seed_vectors(repository, {"a": [1.0, 0.0]})

    with pytest.raises(RetrievalError, match="limit must be a positive integer") as error:
        LocalVectorIndex(repository).search(
            [1.0, 0.0], ["doc1"], limit=limit, min_score=0.0
        )

    assert error.value.code == "invalid_result_limit"
