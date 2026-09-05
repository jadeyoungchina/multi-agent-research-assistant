import asyncio
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from multipart.exceptions import MultipartParseError
from starlette import formparsers
from starlette.requests import Request

from app.api.dependencies import ApplicationServices
from app.api.routes.documents import read_upload_limited
from app.api.uploads import parse_upload_form
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


def test_document_upload_openapi_keeps_multipart_file_controls(client) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/documents"]["post"]

    assert "requestBody" in operation
    body = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    if "$ref" in body:
        body = schema["components"]["schemas"][body["$ref"].split("/")[-1]]
    files = body["properties"]["files"]
    assert files["type"] == "array"
    assert {"type": "string", "format": "binary"} in files["items"].get(
        "anyOf", [files["items"]]
    )


def test_upload_rejects_stream_over_limit(client, document_service) -> None:
    client.app.state.settings.max_upload_total_bytes = 4096
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


def multipart_part(
    name: str, content: bytes, filename: str | None = None, media_type: str = "text/plain"
) -> bytes:
    disposition = f'Content-Disposition: form-data; name="{name}"'
    if filename is not None:
        disposition += f'; filename="{filename}"'
    return (
        f"--upload-boundary\r\n{disposition}\r\nContent-Type: {media_type}\r\n\r\n".encode()
        + content
        + b"\r\n"
    )


def post_stream(
    client,
    body: bytes,
    *,
    chunk_size: int = 64,
    content_length: str | None = None,
    stream_error: BaseException | None = None,
):
    """Exercise the ASGI receive boundary; HTTP clients may buffer the body first."""
    chunks = [body[index : index + chunk_size] for index in range(0, len(body), chunk_size)]
    received = 0
    sent = []

    async def receive():
        nonlocal received
        if received == len(chunks):
            if stream_error is not None:
                raise stream_error
            return {"type": "http.disconnect"}
        chunk = chunks[received]
        received += 1
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": received < len(chunks) or stream_error is not None,
        }

    async def send(message):
        sent.append(message)

    headers = [
        (b"content-type", b"multipart/form-data; boundary=upload-boundary"),
        (b"x-request-id", b"stream-test"),
    ]
    if content_length is not None:
        headers.append((b"content-length", content_length.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "POST",
        "path": "/api/documents",
        "raw_path": b"/api/documents",
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(client.app(scope, receive, send))
    response_start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return SimpleNamespace(
        status=response_start["status"],
        body=json.loads(response_body),
        received_bytes=sum(map(len, chunks[:received])),
        receive_count=received,
        chunk_count=len(chunks),
    )


def assert_stream_rejected(result, document_service, code: str) -> None:
    assert result.status == 413
    assert result.body["error"]["code"] == code
    assert result.body["error"]["request_id"] == "stream-test"
    assert result.receive_count < result.chunk_count
    assert document_service.ingest_calls == []


def test_malformed_multipart_charset_returns_stable_bad_request(
    client, document_service
) -> None:
    body = multipart_part("files", b"valid", "first.txt") + b"--upload-boundary--\r\n"

    response = client.post(
        "/api/documents",
        content=body,
        headers={
            "Content-Type": "multipart/form-data; boundary=upload-boundary; charset=undefined",
            "X-Request-ID": "malformed-charset",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "http_error",
            "message": "There was an error parsing the body",
            "request_id": "malformed-charset",
            "details": {},
        }
    }
    assert response.headers["X-Request-ID"] == "malformed-charset"
    assert document_service.ingest_calls == []


@pytest.mark.parametrize("field_name", ["files", "unexpected"])
def test_streaming_file_limit_stops_receiving_before_ingesting_any_file(
    client, document_service, field_name
) -> None:
    client.app.state.settings.max_upload_file_bytes = 128
    client.app.state.settings.max_upload_total_bytes = 8192
    body = (
        multipart_part("files", b"valid", "first.txt")
        + multipart_part(field_name, b"x" * 4096, "large.txt")
        + b"--upload-boundary--\r\n"
    )

    result = post_stream(client, body)

    assert_stream_rejected(result, document_service, "upload_too_large")
    assert result.received_bytes <= 512


@pytest.mark.parametrize("content_length", [None, "1", "999999"])
def test_streaming_total_limit_counts_actual_bytes_without_trusting_content_length(
    client, document_service, content_length
) -> None:
    client.app.state.settings.max_upload_file_bytes = 8192
    client.app.state.settings.max_upload_total_bytes = 512
    body = multipart_part("files", b"x" * 4096, "large.txt") + b"--upload-boundary--\r\n"

    result = post_stream(client, body, content_length=content_length)

    assert_stream_rejected(result, document_service, "upload_total_too_large")
    assert result.received_bytes == 576


@pytest.mark.parametrize("extra", ["file", "field", "header", "epilogue"])
def test_streaming_total_limit_includes_unexpected_parts_and_multipart_overhead(
    client, document_service, extra
) -> None:
    client.app.state.settings.max_upload_file_bytes = 8192
    client.app.state.settings.max_upload_total_bytes = 512
    body = multipart_part("files", b"valid", "first.txt")
    if extra == "file":
        body += multipart_part("unexpected", b"x" * 4096, "ignored.bin")
    elif extra == "field":
        body += multipart_part("unexpected", b"x" * 4096)
    elif extra == "header":
        body += b"--upload-boundary\r\nX-Ignored: " + b"x" * 4096 + b"\r\n"
        body += b'Content-Disposition: form-data; name="unexpected"\r\n\r\nvalue\r\n'
    body += b"--upload-boundary--\r\n"
    if extra == "epilogue":
        body += b"x" * 4096

    result = post_stream(client, body)

    assert_stream_rejected(result, document_service, "upload_total_too_large")
    assert result.received_bytes == 576


def test_chunked_upload_accepts_exact_limits_and_preserves_files(client, document_service) -> None:
    body = (
        multipart_part("files", b"alpha", "a.txt")
        + multipart_part("ignored", b"small", "ignored.bin")
        + multipart_part("files", b"# beta", "b.md", "text/markdown")
        + b"--upload-boundary--\r\n"
    )
    client.app.state.settings.max_upload_file_bytes = 6
    client.app.state.settings.max_upload_total_bytes = len(body)

    result = post_stream(client, body, chunk_size=7, content_length="999999")

    assert result.status == 201
    assert result.received_bytes == len(body)
    assert document_service.ingest_calls == [
        ("a.txt", "text/plain", b"alpha"),
        ("b.md", "text/markdown", b"# beta"),
    ]
    assert len(result.body["documents"]) == 2


@pytest.fixture
def upload_handles(monkeypatch):
    opened = []
    create_file = formparsers.SpooledTemporaryFile

    def track_file(*args, **kwargs):
        handle = create_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", track_file)
    yield opened
    for handle in opened:
        handle.close()


@pytest.mark.parametrize("error_type", [UnicodeError, MultipartParseError])
def test_upload_context_preserves_consumer_errors_and_closes_files(
    upload_handles, error_type
) -> None:
    body = (
        multipart_part("files", b"valid", "first.txt")
        + multipart_part("ignored", b"small", "ignored.bin")
        + b"--upload-boundary--\r\n"
    )

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "headers": [(b"content-type", b"multipart/form-data; boundary=upload-boundary")],
        },
        receive,
    )
    consumer_error = error_type("endpoint failure after parsing")

    async def fail_after_parsing():
        async with parse_upload_form(request, 1024, 1024):
            raise consumer_error

    with pytest.raises(Exception) as caught:
        asyncio.run(fail_after_parsing())

    assert caught.value is consumer_error
    assert len(upload_handles) == 2
    assert all(handle.closed for handle in upload_handles)


@pytest.mark.parametrize("ending", ["success", "limit", "malformed", "truncated", "disconnect", "cancelled"])
def test_streaming_parser_closes_all_files_on_every_exit(
    client, document_service, upload_handles, ending
) -> None:
    client.app.state.settings.max_upload_file_bytes = 8192
    client.app.state.settings.max_upload_total_bytes = 512
    body = multipart_part("files", b"valid", "first.txt")
    body += multipart_part("ignored", b"small", "ignored.bin")
    stream_error = None
    if ending == "success":
        body += b"--upload-boundary--\r\n"
    elif ending == "limit":
        body += multipart_part("files", b"x" * 4096, "large.txt")
        body += b"--upload-boundary--\r\n"
    elif ending == "malformed":
        body += b"--upload-boundary\r\nInvalid header\r\n\r\n"
    elif ending in {"disconnect", "cancelled"}:
        from starlette.requests import ClientDisconnect

        stream_error = ClientDisconnect() if ending == "disconnect" else asyncio.CancelledError()

    if ending == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            post_stream(client, body, stream_error=stream_error)
    else:
        result = post_stream(client, body, stream_error=stream_error)
        assert result.status == (201 if ending == "success" else 413 if ending == "limit" else 400)
    assert upload_handles
    assert all(handle.closed for handle in upload_handles)
    if ending != "success":
        assert document_service.ingest_calls == []
