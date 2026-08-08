from pathlib import Path

from app.domain.errors import DocumentError
from app.retrieval.loaders import SUPPORTED


class LocalDocumentStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(self, document_id: str, filename: str, content: bytes) -> Path:
        suffix = Path(filename).suffix.casefold()
        if suffix not in SUPPORTED:
            raise DocumentError(
                "unsupported_document_type", f"unsupported extension: {suffix or '<none>'}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        safe_document_id = Path(document_id).name
        target = self.root / f"{safe_document_id}{suffix}"
        resolved_target = target.resolve()
        if not resolved_target.is_relative_to(self.root):
            raise DocumentError("unsafe_storage_path", "storage path escapes upload root")
        target.write_bytes(content)
        return resolved_target

    def delete(self, path: Path) -> bool:
        candidate = path.resolve()
        if not candidate.is_relative_to(self.root):
            return False
        try:
            candidate.unlink()
        except FileNotFoundError:
            return False
        return True
