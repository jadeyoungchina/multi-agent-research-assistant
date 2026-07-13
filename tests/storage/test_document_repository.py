from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.domain.documents import DocumentRecord, EvidenceChunk
from app.storage.database import Database
from app.storage.documents import DocumentRepository


@pytest.fixture
def database(tmp_path: Path) -> Database:
    value = Database(tmp_path / "app.db")
    value.initialize()
    return value


@pytest.fixture
def document_record() -> DocumentRecord:
    return DocumentRecord(
        id="doc-1",
        filename="研究.pdf",
        media_type="application/pdf",
        sha256="sha-1",
        storage_path="uploads/研究.pdf",
        status="processing",
        page_count=0,
        created_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
    )


def _chunk(document: DocumentRecord, chunk_id: str = "chunk-1") -> EvidenceChunk:
    return EvidenceChunk(
        id=chunk_id,
        document_id=document.id,
        filename=document.filename,
        page_number=1,
        chunk_index=0,
        content="evidence",
        content_sha256=f"hash-{chunk_id}",
        embedding_model="fake-hash-3",
    )


def test_document_crud_round_trip(database: Database, document_record: DocumentRecord) -> None:
    repository = DocumentRepository(database)
    repository.add_document(document_record)

    assert repository.get_document(document_record.id) == document_record
    assert repository.get_document_by_sha256(document_record.sha256) == document_record
    assert repository.list_documents() == [document_record]

    updated = document_record.model_copy(
        update={"storage_path": "uploads/final.pdf", "page_count": 2}
    )
    repository.update_document(updated)
    repository.update_status(updated.id, "ready", 3)

    assert repository.get_document(updated.id) == updated.model_copy(
        update={"status": "ready", "page_count": 3, "error_message": None}
    )

    repository.delete_document(updated.id)
    assert repository.get_document(updated.id) is None


def test_chunk_embedding_round_trip(
    database: Database, document_record: DocumentRecord
) -> None:
    repository = DocumentRepository(database)
    repository.add_document(document_record)
    chunk = _chunk(document_record)

    repository.replace_chunks(document_record.id, [(chunk, [0.1, 0.2, 0.3])])

    stored = repository.list_chunks([document_record.id])
    assert stored[0][0] == chunk
    assert np.allclose(stored[0][1], [0.1, 0.2, 0.3])
    assert repository.list_chunks([]) == []


def test_complete_ingestion_replaces_chunks_and_marks_document_ready(
    database: Database, document_record: DocumentRecord
) -> None:
    repository = DocumentRepository(database)
    repository.add_document(document_record)
    repository.replace_chunks(document_record.id, [(_chunk(document_record, "old"), [1.0])])
    ready = document_record.model_copy(
        update={"status": "ready", "page_count": 4, "error_message": None}
    )
    replacement = _chunk(document_record, "new")

    repository.complete_ingestion(ready, [(replacement, [2.0, 3.0])])

    assert repository.get_document(document_record.id) == ready
    assert repository.list_chunks([document_record.id])[0][0] == replacement


def test_complete_ingestion_rolls_back_document_and_chunks_on_failure(
    database: Database,
    document_record: DocumentRecord,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = DocumentRepository(database)
    repository.add_document(document_record)
    old_chunk = _chunk(document_record, "old")
    repository.replace_chunks(document_record.id, [(old_chunk, [1.0])])
    original_insert = repository._insert_chunks

    def insert_then_fail(connection, chunks) -> None:
        original_insert(connection, chunks[:1])
        raise RuntimeError("injected chunk failure")

    monkeypatch.setattr(repository, "_insert_chunks", insert_then_fail)
    ready = document_record.model_copy(update={"status": "ready", "page_count": 2})

    with pytest.raises(RuntimeError, match="injected chunk failure"):
        repository.complete_ingestion(
            ready,
            [(_chunk(document_record, "new"), [2.0])],
        )

    assert repository.get_document(document_record.id) == document_record
    assert repository.list_chunks([document_record.id])[0][0] == old_chunk


def test_fail_ingestion_removes_partial_chunks_and_records_error(
    database: Database, document_record: DocumentRecord
) -> None:
    repository = DocumentRepository(database)
    repository.add_document(document_record)
    repository.replace_chunks(document_record.id, [(_chunk(document_record), [1.0])])

    repository.fail_ingestion(document_record.id, "bad PDF")

    failed = repository.get_document(document_record.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_message == "bad PDF"
    assert repository.list_chunks([document_record.id]) == []
