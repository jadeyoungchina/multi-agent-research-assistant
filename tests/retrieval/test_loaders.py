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
