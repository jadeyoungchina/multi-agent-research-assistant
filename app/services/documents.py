from hashlib import sha256
from pathlib import Path

from app.config import Settings
from app.domain.documents import DocumentRecord
from app.domain.errors import DocumentError
from app.domain.providers import EmbeddingProvider
from app.retrieval.chunking import chunk_pages
from app.retrieval.embedding import embed_chunks
from app.retrieval.loaders import load_document, validate_document_type
from app.storage.documents import DocumentRepository
from app.storage.files import LocalDocumentStore


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        file_store: LocalDocumentStore,
        embedding_provider: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.file_store = file_store
        self.embedding_provider = embedding_provider
        self.settings = settings

    def ingest(self, filename: str, media_type: str, content: bytes) -> DocumentRecord:
        if len(content) > self.settings.max_upload_file_bytes:
            raise DocumentError(
                "document_too_large", "document exceeds configured size limit"
            )
        validate_document_type(filename, media_type)

        digest = sha256(content).hexdigest()
        existing = self.repository.get_document_by_sha256(digest)
        if existing is not None and existing.status == "ready":
            return existing
        if existing is not None and existing.status == "processing":
            raise DocumentError(
                "document_ingestion_in_progress",
                "identical document is already being processed",
            )

        if existing is None:
            document = DocumentRecord.new(filename, media_type, digest)
        else:
            document = existing.model_copy(
                update={
                    "filename": filename,
                    "media_type": media_type,
                    "status": "processing",
                    "page_count": 0,
                    "error_message": None,
                }
            )

        stored_path = self.file_store.save(document.id, filename, content)
        document = document.model_copy(update={"storage_path": str(stored_path)})
        if existing is None:
            self.repository.add_document(document)
        else:
            self.repository.update_document(document)

        try:
            pages = load_document(content, filename, media_type)
            chunks = chunk_pages(
                document.id,
                filename,
                pages,
                self.settings.chunk_size,
                self.settings.chunk_overlap,
            )
            indexed, _embedding_metrics = embed_chunks(
                chunks,
                self.embedding_provider,
                batch_size=self.settings.embedding_batch_size,
            )
            document = document.model_copy(
                update={"status": "ready", "page_count": len(pages)}
            )
            self.repository.complete_ingestion(document, indexed)
            return document
        except Exception as exc:
            self.repository.fail_ingestion(document.id, type(exc).__name__)
            raise

    def list_documents(self) -> list[DocumentRecord]:
        return self.repository.list_documents()

    def delete_document(self, document_id: str) -> None:
        document = self.repository.get_document(document_id)
        if document is None:
            raise DocumentError("document_not_found", "document does not exist")

        self.file_store.delete(Path(document.storage_path))
        self.repository.delete_document(document_id)
