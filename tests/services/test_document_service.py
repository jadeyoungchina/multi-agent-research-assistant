from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import Barrier
from typing import Sequence

import pytest

from app.config import Settings
from app.domain.documents import DocumentRecord, EvidenceChunk
from app.domain.errors import DocumentError, RetrievalError
from app.domain.providers import ProviderMetadata
from app.providers.fake import DeterministicEmbeddingProvider
from app.services.documents import DocumentService
from app.storage.database import Database
from app.storage.documents import DocumentRepository
from app.storage.files import LocalDocumentStore


class CountingEmbeddingProvider(DeterministicEmbeddingProvider):
    def __init__(self, dimensions: int = 8) -> None:
        super().__init__(dimensions)
        self.document_calls = 0

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], ProviderMetadata]:
        self.document_calls += 1
        return super().embed_documents(texts)


class FailingEmbeddingProvider:
    model_name = "failing-model"

    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], ProviderMetadata]:
        raise RetrievalError("provider_failure", "embedding failed")


class RacingDocumentRepository(DocumentRepository):
    def __init__(self, database: Database, lookup_barrier: Barrier) -> None:
        super().__init__(database)
        self.lookup_barrier = lookup_barrier

    def get_document_by_sha256(self, sha256: str) -> DocumentRecord | None:
        document = super().get_document_by_sha256(sha256)
        if document is None:
            self.lookup_barrier.wait(timeout=5)
        return document


@pytest.fixture
def repository(tmp_path: Path) -> DocumentRepository:
    database = Database(tmp_path / "app.db")
    database.initialize()
    return DocumentRepository(database)


@pytest.fixture
def file_store(tmp_path: Path) -> LocalDocumentStore:
    return LocalDocumentStore(tmp_path / "uploads")


@pytest.fixture
def embedding_provider() -> CountingEmbeddingProvider:
    return CountingEmbeddingProvider()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        database_path=tmp_path / "app.db",
        max_upload_file_bytes=1024,
        chunk_size=100,
        chunk_overlap=10,
        embedding_batch_size=2,
    )


@pytest.fixture
def service(
    repository: DocumentRepository,
    file_store: LocalDocumentStore,
    embedding_provider: CountingEmbeddingProvider,
    settings: Settings,
) -> DocumentService:
    return DocumentService(repository, file_store, embedding_provider, settings)


def test_ingest_persists_ready_document_chunks_and_file(
    service: DocumentService, repository: DocumentRepository
) -> None:
    document = service.ingest("notes.md", "text/markdown", b"Alpha\n\nBeta")

    assert document.status == "ready"
    assert document.page_count == 1
    assert repository.get_document(document.id) == document
    chunks = repository.list_chunks([document.id])
    assert chunks
    assert all(chunk.document_id == document.id for chunk, _ in chunks)
    assert all(chunk.filename == "notes.md" for chunk, _ in chunks)
    assert Path(document.storage_path).read_bytes() == b"Alpha\n\nBeta"


def test_identical_ready_content_is_idempotent(
    service: DocumentService, embedding_provider: CountingEmbeddingProvider
) -> None:
    first = service.ingest("first.md", "text/markdown", b"same")

    second = service.ingest("second.md", "text/markdown", b"same")

    assert second == first
    assert embedding_provider.document_calls == 1


def test_identical_processing_content_is_rejected(
    service: DocumentService, repository: DocumentRepository
) -> None:
    content = b"same"
    document = DocumentRecord.new(
        "first.md", "text/markdown", sha256(content).hexdigest()
    )
    repository.add_document(document)

    with pytest.raises(DocumentError) as raised:
        service.ingest("second.md", "text/markdown", content)

    assert raised.value.code == "document_ingestion_in_progress"
    assert repository.get_document(document.id) == document


def test_identical_failed_content_retries_same_document(
    service: DocumentService,
    repository: DocumentRepository,
    embedding_provider: CountingEmbeddingProvider,
) -> None:
    content = b"retry me"
    service.embedding_provider = FailingEmbeddingProvider()
    with pytest.raises(RetrievalError):
        service.ingest("first.md", "text/markdown", content)
    failed = repository.list_documents()[0]

    service.embedding_provider = embedding_provider
    ready = service.ingest("second.md", "text/markdown", content)

    assert ready.id == failed.id
    assert ready.filename == "second.md"
    assert ready.status == "ready"
    assert len(repository.list_documents()) == 1
    assert embedding_provider.document_calls == 1


def test_failed_retry_with_new_extension_removes_superseded_upload(
    service: DocumentService, repository: DocumentRepository
) -> None:
    content = b"retry me"
    service.embedding_provider = FailingEmbeddingProvider()
    with pytest.raises(RetrievalError):
        service.ingest("first.txt", "text/plain", content)
    failed = repository.list_documents()[0]
    failed_path = Path(failed.storage_path)

    service.embedding_provider = CountingEmbeddingProvider()
    ready = service.ingest("second.md", "text/markdown", content)

    ready_path = Path(ready.storage_path)
    assert ready_path != failed_path
    assert ready_path.read_bytes() == content
    assert not failed_path.exists()

    service.delete_document(ready.id)
    assert not ready_path.exists()


def test_concurrent_identical_uploads_do_not_leak_files_or_sqlite_errors(
    tmp_path: Path, settings: Settings
) -> None:
    database = Database(tmp_path / "race.db")
    database.initialize()
    repository = RacingDocumentRepository(database, Barrier(2))
    file_store = LocalDocumentStore(tmp_path / "race-uploads")
    service = DocumentService(
        repository, file_store, CountingEmbeddingProvider(), settings
    )

    def ingest() -> DocumentRecord:
        return service.ingest("notes.txt", "text/plain", b"same content")

    results: list[DocumentRecord] = []
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(ingest) for _ in range(2)]
        for future in futures:
            try:
                results.append(future.result(timeout=10))
            except Exception as exc:
                errors.append(exc)

    documents = repository.list_documents()
    assert len(documents) == 1
    assert results
    assert all(result.id == documents[0].id for result in results)
    assert all(
        isinstance(error, DocumentError)
        and error.code == "document_ingestion_in_progress"
        for error in errors
    ), errors
    assert len(list(file_store.root.iterdir())) == 1


def test_provider_failure_marks_document_failed_without_chunks_and_retains_file(
    service: DocumentService, repository: DocumentRepository
) -> None:
    content = b"sensitive upload bytes"
    service.embedding_provider = FailingEmbeddingProvider()

    with pytest.raises(RetrievalError):
        service.ingest("notes.txt", "text/plain", content)

    failed = repository.list_documents()[0]
    assert failed.status == "failed"
    assert failed.error_message == "RetrievalError"
    assert repository.list_chunks([failed.id]) == []
    assert Path(failed.storage_path).read_bytes() == content
    assert content.decode() not in (failed.error_message or "")


def test_failed_retry_removes_preexisting_partial_chunks(
    service: DocumentService, repository: DocumentRepository
) -> None:
    content = b"retry me"
    document = DocumentRecord.new(
        "notes.txt", "text/plain", sha256(content).hexdigest()
    ).model_copy(update={"status": "failed", "error_message": "old failure"})
    repository.add_document(document)
    partial = EvidenceChunk(
        id="partial",
        document_id=document.id,
        filename=document.filename,
        page_number=None,
        chunk_index=0,
        content="partial",
        content_sha256="partial-digest",
        embedding_model="old-model",
    )
    repository.replace_chunks(document.id, [(partial, [1.0])])
    service.embedding_provider = FailingEmbeddingProvider()

    with pytest.raises(RetrievalError):
        service.ingest("notes.txt", "text/plain", content)

    assert repository.list_chunks([document.id]) == []
    assert repository.get_document(document.id).status == "failed"  # type: ignore[union-attr]


def test_upload_size_is_rejected_before_disk_write(
    service: DocumentService,
    repository: DocumentRepository,
    file_store: LocalDocumentStore,
    settings: Settings,
) -> None:
    settings.max_upload_file_bytes = 3

    with pytest.raises(DocumentError) as raised:
        service.ingest("notes.txt", "text/plain", b"four")

    assert raised.value.code == "document_too_large"
    assert repository.list_documents() == []
    assert not file_store.root.exists()


def test_invalid_document_type_is_rejected_before_disk_write(
    service: DocumentService,
    repository: DocumentRepository,
    file_store: LocalDocumentStore,
) -> None:
    with pytest.raises(DocumentError) as raised:
        service.ingest("notes.exe", "application/octet-stream", b"binary")

    assert raised.value.code == "unsupported_document_type"
    assert repository.list_documents() == []
    assert not file_store.root.exists()


def test_empty_file_is_failed_and_retained_for_cleanup(
    service: DocumentService, repository: DocumentRepository
) -> None:
    with pytest.raises(DocumentError) as raised:
        service.ingest("empty.txt", "text/plain", b"")

    assert raised.value.code == "empty_document"
    failed = repository.list_documents()[0]
    assert failed.status == "failed"
    assert failed.error_message == "DocumentError"
    assert repository.list_chunks([failed.id]) == []
    assert Path(failed.storage_path).exists()
    assert Path(failed.storage_path).read_bytes() == b""


def test_list_documents_returns_repository_records(service: DocumentService) -> None:
    first = service.ingest("first.txt", "text/plain", b"first")
    second = service.ingest("second.txt", "text/plain", b"second")

    assert service.list_documents() == [first, second]


def test_delete_document_removes_database_chunks_and_stored_bytes(
    service: DocumentService, repository: DocumentRepository
) -> None:
    document = service.ingest("notes.txt", "text/plain", b"delete me")
    stored_path = Path(document.storage_path)

    service.delete_document(document.id)

    assert repository.get_document(document.id) is None
    assert repository.list_chunks([document.id]) == []
    assert not stored_path.exists()


def test_delete_document_succeeds_when_stored_file_is_already_missing(
    service: DocumentService, repository: DocumentRepository
) -> None:
    document = service.ingest("notes.txt", "text/plain", b"delete me")
    Path(document.storage_path).unlink()

    service.delete_document(document.id)

    assert repository.get_document(document.id) is None


def test_delete_unknown_document_raises_stable_domain_error(
    service: DocumentService,
) -> None:
    with pytest.raises(DocumentError) as raised:
        service.delete_document("unknown-id")

    assert raised.value.code == "document_not_found"
