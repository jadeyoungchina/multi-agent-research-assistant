from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request
from multipart.multipart import parse_options_header
from starlette.datastructures import FormData, Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import FormParser, MultiPartException, MultiPartParser


async def _limited_stream(request: Request, total_limit: int) -> AsyncGenerator[bytes, None]:
    received_bytes = 0
    async for chunk in request.stream():
        # Count every body byte before the parser can buffer it, including headers,
        # boundaries, unexpected fields and epilogue. Content-Length is untrusted.
        received_bytes += len(chunk)
        if received_bytes > total_limit:
            raise HTTPException(status_code=413, detail={"code": "upload_total_too_large"})
        yield chunk


class _LimitedMultiPartParser(MultiPartParser):
    def __init__(
        self, headers: Headers, stream: AsyncGenerator[bytes, None], per_file_limit: int
    ) -> None:
        super().__init__(headers, stream)
        self.per_file_limit = per_file_limit
        self.part_bytes = 0
        self.complete = False

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self.part_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            # UploadFile.size is updated later, when Starlette flushes this chunk.
            # Count callbacks instead, including files under unexpected field names.
            self.part_bytes += end - start
            if self.part_bytes > self.per_file_limit:
                raise HTTPException(status_code=413, detail={"code": "upload_too_large"})
        super().on_part_data(data, start, end)

    def on_end(self) -> None:
        self.complete = True

    def close(self) -> None:
        # Starlette's registry also owns unexpected and still-incomplete files.
        # Its built-in cleanup only handles MultiPartException, not limits,
        # parser errors, disconnects or cancellation. Always release the registry.
        for handle in self._files_to_close_on_error:
            handle.close()


@asynccontextmanager
async def parse_upload_form(
    request: Request, per_file_limit: int, total_limit: int
) -> AsyncIterator[FormData]:
    parser = None
    try:
        # Match FastAPI's malformed-body handling only during parsing. Exceptions
        # from the consumer after yield must propagate without being reclassified.
        try:
            content_type, _ = parse_options_header(request.headers.get("Content-Type"))
            stream = _limited_stream(request, total_limit)
            if content_type == b"multipart/form-data":
                parser = _LimitedMultiPartParser(request.headers, stream, per_file_limit)
                form = await parser.parse()
                # python-multipart 0.0.9 finalize() does not validate the final boundary.
                if not parser.complete:
                    raise HTTPException(status_code=400, detail="There was an error parsing the body")
            elif content_type == b"application/x-www-form-urlencoded":
                form = await FormParser(request.headers, stream).parse()
            else:
                form = FormData()
        except StarletteHTTPException:
            raise
        except MultiPartException as exc:
            raise HTTPException(status_code=400, detail=exc.message) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail="There was an error parsing the body") from exc
        yield form
    finally:
        if parser is not None:
            parser.close()
