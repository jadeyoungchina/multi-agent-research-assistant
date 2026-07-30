import logging
import threading

import pytest

from app.domain.errors import DocumentError
from app.retrieval import load_document as exported_load_document
from app.retrieval.loaders import load_document, validate_document_type


@pytest.mark.parametrize(
    ("filename", "media_type", "expected_kind"),
    [
        ("paper.pdf", "application/pdf", "pdf"),
        ("notes.md", "text/markdown", "markdown"),
        ("notes.markdown", "text/markdown", "markdown"),
        ("notes.txt", "text/plain", "text"),
    ],
)
def test_validate_document_type_accepts_supported_pairs(
    filename: str, media_type: str, expected_kind: str
) -> None:
    assert validate_document_type(filename, media_type) == expected_kind


def test_validate_document_type_rejects_mismatched_mime() -> None:
    with pytest.raises(DocumentError) as raised:
        validate_document_type("paper.pdf", "text/plain")
    assert raised.value.code == "document_type_mismatch"


def test_retrieval_package_exports_document_loader() -> None:
    assert exported_load_document is load_document


def test_markdown_loader_normalizes_newlines_and_nuls() -> None:
    pages = load_document(b"Heading\r\n\x00Body   \r\n\r\n\r\nEnd", "notes.md", "text/markdown")

    assert pages[0].page_number is None
    assert pages[0].text == "Heading\nBody\n\nEnd"


def test_text_loader_rejects_invalid_utf8() -> None:
    with pytest.raises(DocumentError) as raised:
        load_document(b"\xff\xfe", "notes.txt", "text/plain")
    assert raised.value.code == "document_decode_failed"


def test_pdf_loader_does_not_log_uploaded_bytes_on_parse_failure(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = "M4RKR"
    unrelated_marker = "unrelated application warning"
    caplog.set_level(logging.WARNING, logger="pypdf")
    logging.getLogger("app.test").warning(unrelated_marker)

    with pytest.raises(DocumentError) as raised:
        load_document(marker.encode("utf-8"), "paper.pdf", "application/pdf")

    captured = capsys.readouterr()
    assert raised.value.code == "document_parse_failed"
    assert marker not in "\n".join(record.getMessage() for record in caplog.records)
    assert marker not in captured.err
    assert unrelated_marker in "\n".join(record.getMessage() for record in caplog.records)


def test_pdf_loader_filters_last_resort_without_configured_handlers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "L4STR"
    root_logger = logging.getLogger()
    pypdf_loggers = [
        logger
        for name, logger in logging.Logger.manager.loggerDict.items()
        if (name == "pypdf" or name.startswith("pypdf."))
        and isinstance(logger, logging.Logger)
    ]
    handler_state = [(root_logger, list(root_logger.handlers))] + [
        (logger, list(logger.handlers)) for logger in pypdf_loggers
    ]
    reader_logger = logging.getLogger("pypdf._reader")
    original_disabled = reader_logger.disabled
    original_last_resort = logging.lastResort

    class CapturingLastResort(logging.StreamHandler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)
            super().emit(record)

    fallback = CapturingLastResort()
    try:
        for logger, _ in handler_state:
            logger.handlers.clear()
        reader_logger.disabled = False
        logging.lastResort = fallback

        with pytest.raises(DocumentError) as raised:
            load_document(marker.encode("utf-8"), "paper.pdf", "application/pdf")

        captured = capsys.readouterr()
        assert raised.value.code == "document_parse_failed"
        assert marker not in "\n".join(record.getMessage() for record in fallback.records)
        assert marker not in captured.err
        assert fallback.filters == []
    finally:
        logging.lastResort = original_last_resort
        reader_logger.disabled = original_disabled
        for logger, handlers in handler_state:
            logger.handlers[:] = handlers


def test_pdf_loader_does_not_log_uploaded_bytes_during_page_traversal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "P4G3M"

    class FakePage:
        def extract_text(self) -> str:
            logging.getLogger("pypdf.generic._data_structures").warning(marker)
            return "page text"

    class FakeReader:
        def __init__(self, stream: object) -> None:
            self.pages = [FakePage()]

    monkeypatch.setattr("app.retrieval.loaders.PdfReader", FakeReader)
    caplog.set_level(logging.WARNING)

    pages = load_document(b"PDF", "paper.pdf", "application/pdf")

    captured = capsys.readouterr()
    assert pages[0].text == "page text"
    assert marker not in "\n".join(record.getMessage() for record in caplog.records)
    assert marker not in captured.err


def test_pdf_loader_restores_pypdf_logging_after_page_extraction_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    post_failure_marker = "pypdf logging restored"
    reader_logger = logging.getLogger("pypdf._reader")
    original_disabled = reader_logger.disabled

    class FailingPage:
        def extract_text(self) -> str:
            raise RuntimeError("page extraction failed")

    class FakeReader:
        def __init__(self, stream: object) -> None:
            self.pages = [FailingPage()]

    monkeypatch.setattr("app.retrieval.loaders.PdfReader", FakeReader)
    caplog.set_level(logging.WARNING)
    reader_logger.disabled = False
    try:
        with pytest.raises(DocumentError) as raised:
            load_document(b"PDF", "paper.pdf", "application/pdf")
        reader_logger.warning(post_failure_marker)
    finally:
        reader_logger.disabled = original_disabled

    assert raised.value.code == "document_parse_failed"
    assert post_failure_marker in "\n".join(record.getMessage() for record in caplog.records)


def test_pdf_loader_keeps_unrelated_application_logging_during_parse(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    marker = "unrelated application log during PDF parse"

    class FakeReader:
        def __init__(self, stream: object) -> None:
            logging.getLogger("app.test").warning(marker)
            self.pages = []

    monkeypatch.setattr("app.retrieval.loaders.PdfReader", FakeReader)
    caplog.set_level(logging.WARNING)

    with pytest.raises(DocumentError) as raised:
        load_document(b"PDF", "paper.pdf", "application/pdf")

    assert raised.value.code == "empty_document"
    assert marker in "\n".join(record.getMessage() for record in caplog.records)


def test_overlapping_pdf_loads_restore_logging_state(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    finished = [threading.Event(), threading.Event()]
    counter_lock = threading.Lock()
    reader_logger = logging.getLogger("pypdf._reader")
    original_disabled = reader_logger.disabled
    call_count = 0
    restored_disabled = True

    class BlockingReader:
        def __init__(self, stream: object) -> None:
            nonlocal call_count
            with counter_lock:
                index = call_count
                call_count += 1
            entered[index].set()
            assert release[index].wait(timeout=5)
            self.pages = []

    def load_in_thread(index: int) -> None:
        try:
            with pytest.raises(DocumentError):
                load_document(b"PDF", "paper.pdf", "application/pdf")
        finally:
            finished[index].set()

    monkeypatch.setattr("app.retrieval.loaders.PdfReader", BlockingReader)
    caplog.set_level(logging.WARNING)
    reader_logger.disabled = False
    first = threading.Thread(target=load_in_thread, args=(0,))
    second = threading.Thread(target=load_in_thread, args=(1,))
    first.start()
    try:
        assert entered[0].wait(timeout=5)
        second.start()
        release[0].set()
        assert finished[0].wait(timeout=5)
        assert entered[1].wait(timeout=5)
        logging.getLogger("pypdf._reader").warning("concurrent pypdf log")
        release[1].set()
        assert finished[1].wait(timeout=5)
    finally:
        restored_disabled = reader_logger.disabled
        release[0].set()
        release[1].set()
        first.join(timeout=5)
        second.join(timeout=5)
        reader_logger.disabled = original_disabled

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "concurrent pypdf log" in messages
    assert restored_disabled is False


def test_pdf_loader_preserves_one_based_page_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        def __init__(self, stream: object) -> None:
            self.pages = [FakePage("First page"), FakePage("Second page")]

    monkeypatch.setattr("app.retrieval.loaders.PdfReader", FakeReader)

    pages = load_document(b"not a real PDF", "paper.pdf", "application/pdf")

    assert [(page.page_number, page.text) for page in pages] == [
        (1, "First page"),
        (2, "Second page"),
    ]


@pytest.mark.parametrize(
    ("content", "filename", "media_type"),
    [
        (b" \r\n\x00\r\n", "notes.md", "text/markdown"),
        (b" \n", "notes.txt", "text/plain"),
    ],
)
def test_loader_rejects_document_without_extractable_text(
    content: bytes, filename: str, media_type: str
) -> None:
    with pytest.raises(DocumentError) as raised:
        load_document(content, filename, media_type)
    assert raised.value.code == "empty_document"
