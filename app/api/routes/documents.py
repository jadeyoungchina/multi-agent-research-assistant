from collections.abc import Sequence

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.api.dependencies import ApplicationServices, get_services
from app.api.schemas import DocumentResponse, DocumentUploadResponse
from app.domain.documents import DocumentRecord
from app.retrieval.loaders import validate_document_type


router = APIRouter(prefix="/api/documents", tags=["documents"])
_READ_CHUNK_BYTES = 64 * 1024


async def read_upload_limited(
    upload: UploadFile, per_file_limit: int, remaining: int
) -> bytes:
    data = bytearray()
    try:
        while chunk := await upload.read(_READ_CHUNK_BYTES):
            data.extend(chunk)
            if len(data) > per_file_limit:
                raise HTTPException(status_code=413, detail={"code": "upload_too_large"})
            if len(data) > remaining:
                raise HTTPException(
                    status_code=413, detail={"code": "upload_total_too_large"}
                )
        return bytes(data)
    finally:
        await upload.close()


def _response_document(document: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        media_type=document.media_type,
        sha256=document.sha256,
        status=document.status,
        page_count=document.page_count,
        created_at=document.created_at,
    )


def _settings(request: Request, services: ApplicationServices) -> object:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = services.documents.settings
    return settings


async def _close_uploads(uploads: Sequence[UploadFile]) -> None:
    for upload in uploads:
        await upload.close()


@router.post("", status_code=201, response_model=DocumentUploadResponse)
async def upload_documents(
    request: Request,
    files: list[UploadFile | str] = File(default=[]),
    services: ApplicationServices = Depends(get_services),
) -> DocumentUploadResponse:
    submitted = files
    uploads = [item for item in submitted if isinstance(item, StarletteUploadFile)]
    try:
        if not submitted:
            raise HTTPException(
                status_code=400, detail={"code": "upload_files_required"}
            )
        if len(uploads) != len(submitted):
            raise HTTPException(status_code=400, detail={"code": "invalid_filename"})

        settings = _settings(request, services)
        prepared: list[tuple[str, str, bytes]] = []
        accepted_bytes = 0
        for upload in uploads:
            filename = (upload.filename or "").strip()
            if not filename:
                raise HTTPException(status_code=400, detail={"code": "invalid_filename"})
            media_type = upload.content_type or ""
            validate_document_type(filename, media_type)
            content = await read_upload_limited(
                upload,
                settings.max_upload_file_bytes,
                settings.max_upload_total_bytes - accepted_bytes,
            )
            accepted_bytes += len(content)
            prepared.append((filename, media_type, content))
    finally:
        await _close_uploads(uploads)

    documents = [
        await run_in_threadpool(services.documents.ingest, filename, media_type, content)
        for filename, media_type, content in prepared
    ]
    return DocumentUploadResponse(
        documents=[_response_document(document) for document in documents]
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    services: ApplicationServices = Depends(get_services),
) -> list[DocumentResponse]:
    documents = await run_in_threadpool(services.documents.list_documents)
    return [_response_document(document) for document in documents]


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str, services: ApplicationServices = Depends(get_services)
) -> None:
    await run_in_threadpool(services.documents.delete_document, document_id)
