import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.dependencies import ApplicationServices
from app.api.routes.documents import read_upload_limited
from app.domain.documents import DocumentRecord


class RecordingDocumentService:
    def __init__(self) -> None:
        self.ingest_calls: list[tuple[str, str, bytes]] = []
        self.documents: list[DocumentRecord] = []
        self.deleted_ids: list[str] = []

    def ingest(self, filename: str, media_type: str, content: bytes) -> DocumentRecord:
        self.ingest_calls.append((filename, media_type, content))
        document = DocumentRecord(
            id=f"doc{len(self.ingest_calls)}",
            filename=filename,
            media_type=media_type,
            sha256="a" * 64,
            storage_path="private/upload/path",
            status="ready",
            page_count=1,
            created_at=datetime(2026, 9, 10, tzinfo=UTC),
        )
        self.documents.append(document)
        return document

    def list_documents(self) -> list[DocumentRecord]:
        return self.documents

    def delete_document(self, document_id: str) -> None:
        self.deleted_ids.append(document_id)


class ChunkedUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.read_sizes: list[int] = []
        self.close_count = 0

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.chunks.pop(0) if self.chunks else b""

    async def close(self) -> None:
        self.close_count += 1


@pytest.fixture
def document_service(client) -> RecordingDocumentService:
    service = RecordingDocumentService()
    client.app.state.services = ApplicationServices(
        documents=service, research=SimpleNamespace()
    )
    client.app.state.settings = SimpleNamespace(
        max_upload_file_bytes=1024,
        max_upload_total_bytes=1024,
    )
    return service


def test_uploads_multiple_documents(client, document_service) -> None:
    response = client.post(
        "/api/documents",
        files=[
            ("files", ("a.txt", b"alpha", "text/plain")),
            ("files", ("b.md", b"# beta", "text/markdown")),
        ],
    )

    assert response.status_code == 201
    assert [call[0] for call in document_service.ingest_calls] == ["a.txt", "b.md"]
    assert len(response.json()["documents"]) == 2


def test_upload_rejects_stream_over_limit(client, document_service) -> None:
    response = client.post(
        "/api/documents",
        files=[("files", ("large.txt", b"x" * 1025, "text/plain"))],
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"
    assert document_service.ingest_calls == []


def test_upload_rejects_combined_request_over_limit(client, document_service) -> None:
    response = client.post(
        "/api/documents",
        files=[
            ("files", ("a.txt", b"a" * 700, "text/plain")),
            ("files", ("b.txt", b"b" * 700, "text/plain")),
        ],
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_total_too_large"
    assert document_service.ingest_calls == []


def test_list_and_delete_documents(client, document_service) -> None:
    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == []
    assert client.delete("/api/documents/doc1").status_code == 204
    assert document_service.deleted_ids == ["doc1"]


def test_upload_rejects_empty_filename_with_stable_code(client, document_service) -> None:
    response = client.post(
        "/api/documents",
        files=[("files", ("", b"content", "text/plain"))],
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_filename"
    assert document_service.ingest_calls == []


def test_upload_rejects_missing_file_list_with_stable_code(client, document_service) -> None:
    response = client.post("/api/documents")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upload_files_required"
    assert document_service.ingest_calls == []


def test_upload_validates_every_file_before_any_ingestion(client, document_service) -> None:
    response = client.post(
        "/api/documents",
        files=[
            ("files", ("valid.txt", b"valid", "text/plain")),
            ("files", ("invalid.txt", b"invalid", "application/pdf")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "document_type_mismatch"
    assert document_service.ingest_calls == []


def test_upload_response_excludes_private_document_fields(client, document_service) -> None:
    response = client.post(
        "/api/documents",
        files=[("files", ("safe.txt", b"safe", "text/plain"))],
    )

    assert response.status_code == 201
    document = response.json()["documents"][0]
    assert set(document) == {
        "id",
        "filename",
        "media_type",
        "sha256",
        "status",
        "page_count",
        "created_at",
    }
    assert "private/upload/path" not in response.text


def test_limited_reader_closes_upload_after_overflow() -> None:
    upload = ChunkedUpload([b"x" * (64 * 1024 + 1)])

    with pytest.raises(HTTPException) as raised:
        asyncio.run(read_upload_limited(upload, 1024, 1024))

    assert raised.value.status_code == 413
    assert raised.value.detail == {"code": "upload_too_large"}
    assert upload.read_sizes == [64 * 1024]
    assert upload.close_count == 1
