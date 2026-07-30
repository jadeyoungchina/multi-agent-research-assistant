from pathlib import Path

import pytest

from app.domain.errors import DocumentError
from app.storage.files import LocalDocumentStore


def test_store_ignores_user_path_components(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)

    saved = store.save("doc123", "../../secret.pdf", b"pdf")

    assert saved == (tmp_path / "doc123.pdf").resolve()
    assert saved.read_bytes() == b"pdf"


def test_store_uses_canonical_validated_suffix(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)

    saved = store.save("doc123", "named.PDF", b"pdf")

    assert saved == (tmp_path / "doc123.pdf").resolve()


def test_store_rejects_an_unsupported_suffix(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)

    with pytest.raises(DocumentError) as raised:
        store.save("doc123", "unsafe.exe", b"binary")

    assert raised.value.code == "unsupported_document_type"


def test_delete_is_confined_to_upload_root(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("keep", encoding="utf-8")

    assert store.delete(outside) is False
    assert outside.exists()
