from hashlib import sha256

from app.domain.documents import EvidenceChunk, LoadedPage


SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", " ")


def make_evidence_id(
    document_id: str,
    page_number: int | None,
    chunk_index: int,
    digest: str,
) -> str:
    """Build a deterministic identifier for a specific piece of document evidence."""
    raw = f"{document_id}:{page_number or 0}:{chunk_index}:{digest}".encode("utf-8")
    return "ev_" + sha256(raw).hexdigest()


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split one page without emitting blank chunks or stalling at a separator."""
    chunks: list[str] = []
    start = 0

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end

        if hard_end < len(text):
            window = text[start:hard_end]
            candidates = [window.rfind(separator) for separator in SEPARATORS]
            boundary = max(candidates)
            if boundary >= chunk_size // 2:
                end = start + boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def chunk_pages(
    document_id: str,
    filename: str,
    pages: list[LoadedPage],
    chunk_size: int,
    overlap: int,
) -> list[EvidenceChunk]:
    """Create globally indexed, page-local evidence chunks for a document."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    result: list[EvidenceChunk] = []
    for page in pages:
        for text in _split_text(page.text, chunk_size, overlap):
            digest = sha256(text.encode("utf-8")).hexdigest()
            chunk_index = len(result)
            result.append(
                EvidenceChunk(
                    id=make_evidence_id(
                        document_id, page.page_number, chunk_index, digest
                    ),
                    document_id=document_id,
                    filename=filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    content=text,
                    content_sha256=digest,
                )
            )

    return result
