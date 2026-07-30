# Adapted from NirDiamant/GenAI_Agents:
# document_intake_agent_langgraph.ipynb cells 13, 15, 19, 23
# Upstream commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46
# Changes: accepts uploaded bytes, adds strict MIME/extension matching,
# local PDF parsing, page metadata, normalization, and no remote conversion.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt

from contextlib import contextmanager
from io import BytesIO
import logging
from pathlib import Path
import re
from threading import RLock, local
from typing import Iterator

from pypdf import PdfReader

from app.domain.documents import LoadedPage
from app.domain.errors import DocumentError

SUPPORTED = {
    ".pdf": ("application/pdf", "pdf"),
    ".md": ("text/markdown", "markdown"),
    ".markdown": ("text/markdown", "markdown"),
    ".txt": ("text/plain", "text"),
}

_PYPDF_LOG_LOCK = RLock()
_PYPDF_LOG_STATE = local()


class _PypdfLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        is_pypdf = record.name == "pypdf" or record.name.startswith("pypdf.")
        return not (is_pypdf and getattr(_PYPDF_LOG_STATE, "depth", 0))


_PYPDF_LOG_FILTER = _PypdfLogFilter()


def validate_document_type(filename: str, media_type: str) -> str:
    suffix = Path(filename).suffix.casefold()
    expected = SUPPORTED.get(suffix)
    if expected is None:
        raise DocumentError(
            "unsupported_document_type", f"unsupported extension: {suffix or '<none>'}"
        )
    if media_type.split(";", 1)[0].strip().casefold() != expected[0]:
        raise DocumentError(
            "document_type_mismatch", "file extension and media type do not match"
        )
    return expected[1]


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _logging_handlers() -> list[logging.Handler]:
    handlers = list(logging.getLogger().handlers)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            handlers.extend(logger.handlers)
    return list(dict.fromkeys(handlers))


@contextmanager
def _suppress_pypdf_logging() -> Iterator[None]:
    """Filter PyPDF records from this parsing thread without muting other logs."""
    if getattr(_PYPDF_LOG_STATE, "depth", 0):
        _PYPDF_LOG_STATE.depth += 1
        try:
            yield
        finally:
            _PYPDF_LOG_STATE.depth -= 1
        return

    with _PYPDF_LOG_LOCK:
        added_to: list[logging.Handler] = []
        try:
            for handler in _logging_handlers():
                if _PYPDF_LOG_FILTER not in handler.filters:
                    handler.addFilter(_PYPDF_LOG_FILTER)
                    added_to.append(handler)
            _PYPDF_LOG_STATE.depth = 1
            yield
        finally:
            _PYPDF_LOG_STATE.depth = 0
            for handler in added_to:
                handler.removeFilter(_PYPDF_LOG_FILTER)


def load_document(content: bytes, filename: str, media_type: str) -> list[LoadedPage]:
    kind = validate_document_type(filename, media_type)
    if kind == "pdf":
        try:
            with _suppress_pypdf_logging():
                reader = PdfReader(BytesIO(content))
                pages = [
                    LoadedPage(page_number=index, text=_normalize(page.extract_text() or ""))
                    for index, page in enumerate(reader.pages, start=1)
                ]
        except Exception as exc:
            raise DocumentError("document_parse_failed", "PDF could not be parsed") from exc
    else:
        try:
            text = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise DocumentError("document_decode_failed", "document must use UTF-8") from exc
        pages = [LoadedPage(page_number=None, text=_normalize(text))]
    pages = [page for page in pages if page.text]
    if not pages:
        raise DocumentError("empty_document", "document contains no extractable text")
    return pages
