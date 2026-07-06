# Document Ingestion and Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely ingest PDF, Markdown, and TXT files, create deterministic source-preserving chunks, persist embeddings, and retrieve fused evidence from the local SQLite index.

**Architecture:** A loader dispatcher converts uploaded bytes into page-aware `LoadedPage` records. A deterministic page-local chunker creates stable Evidence IDs, an embedding stage persists vectors through `DocumentRepository`, and a NumPy cosine index plus reciprocal-rank fusion serves evidence to the later Retriever Agent.

**Tech Stack:** Python 3.11, PyPDF 4.3.1, NumPy 1.26.4, SQLite, Pydantic 2.8.2, pytest 8.3.2.

**Spec:** `docs/superpowers/specs/2026-09-08-multi-agent-research-assistant-design.md`

## Global Constraints

- Accept only `.pdf`, `.md`, `.markdown`, and `.txt` with matching media types.
- Do not call Hushvert, CORE, web search, or any remote URL loader.
- Preserve source filename, one-based PDF page number, global zero-based chunk index, and content hash.
- Use `chunk_size=1200`, `chunk_overlap=200`, `embedding_batch_size=32`,
  `retrieval_top_k=6`, `retrieval_candidate_k=20`, `retrieval_rrf_k=60`,
  `retrieval_min_similarity=0.15`, and `max_query_expansions=4` from `Settings`.
- Chunking never crosses a PDF page boundary.
- Embeddings are normalized before persistence and comparison.
- Uploaded file bytes, full embeddings, and secrets never enter logs.
- Re-ingesting identical bytes returns the existing ready document and does not call the embedding provider again.
- Default tests use `DeterministicEmbeddingProvider` and no network.
- Adapted files carry the approved upstream source header and license pointer.

---

### Task 1: Safe Document Type Routing and Local File Store

**Files:**
- Modify: `app/retrieval/__init__.py`
- Create: `app/retrieval/loaders.py`
- Create: `app/storage/files.py`
- Create: `tests/retrieval/test_loaders.py`
- Create: `tests/storage/test_files.py`

**Interfaces:**
- Consumes: `LoadedPage`, `DocumentError`, `Settings.upload_dir`.
- Produces: `validate_document_type()`, `load_document()`, `LocalDocumentStore`.

- [ ] **Step 1: Write failing loader tests**

```python
# tests/retrieval/test_loaders.py
from io import BytesIO

import pytest

from app.domain.errors import DocumentError
from app.retrieval.loaders import load_document, validate_document_type


@pytest.mark.parametrize(
    ("filename", "media_type"),
    [
        ("paper.pdf", "application/pdf"),
        ("notes.md", "text/markdown"),
        ("notes.markdown", "text/markdown"),
        ("notes.txt", "text/plain"),
    ],
)
def test_validate_document_type_accepts_supported_pairs(filename: str, media_type: str) -> None:
    assert validate_document_type(filename, media_type) in {"pdf", "markdown", "text"}


def test_validate_document_type_rejects_mismatched_mime() -> None:
    with pytest.raises(DocumentError, match="document_type_mismatch"):
        validate_document_type("paper.pdf", "text/plain")


def test_markdown_loader_normalizes_newlines_and_nuls() -> None:
    pages = load_document(b"Heading\r\n\x00Body   \r\n\r\n\r\nEnd", "notes.md", "text/markdown")
    assert pages[0].page_number is None
    assert pages[0].text == "Heading\nBody\n\nEnd"


def test_text_loader_rejects_invalid_utf8() -> None:
    with pytest.raises(DocumentError, match="document_decode_failed"):
        load_document(b"\xff\xfe", "notes.txt", "text/plain")
```

Monkeypatch `PdfReader` with two fake pages whose `extract_text()` values are known, then assert page numbers `1` and `2` are preserved. Add an all-empty-document test that raises `DocumentError("empty_document", ...)`.

- [ ] **Step 2: Run tests and observe import failure**

Run: `.venv\Scripts\python.exe -m pytest tests/retrieval/test_loaders.py -v`

Expected: FAIL because `app.retrieval.loaders` does not exist.

- [ ] **Step 3: Implement the upstream-adapted route and local loaders**

```python
# app/retrieval/loaders.py
# Adapted from NirDiamant/GenAI_Agents:
# document_intake_agent_langgraph.ipynb cells 13, 15, 19, 23
# Upstream commit: 4c95ae14cc2462c442b5c064cccd74430d02bc46
# Changes: accepts uploaded bytes, adds strict MIME/extension matching,
# local PDF parsing, page metadata, normalization, and no remote conversion.
# License: THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt

from io import BytesIO
from pathlib import Path
import re

from pypdf import PdfReader

from app.domain.documents import LoadedPage
from app.domain.errors import DocumentError

SUPPORTED = {
    ".pdf": ("application/pdf", "pdf"),
    ".md": ("text/markdown", "markdown"),
    ".markdown": ("text/markdown", "markdown"),
    ".txt": ("text/plain", "text"),
}


def validate_document_type(filename: str, media_type: str) -> str:
    suffix = Path(filename).suffix.casefold()
    expected = SUPPORTED.get(suffix)
    if expected is None:
        raise DocumentError("unsupported_document_type", f"unsupported extension: {suffix or '<none>'}")
    if media_type.split(";", 1)[0].strip().casefold() != expected[0]:
        raise DocumentError("document_type_mismatch", "file extension and media type do not match")
    return expected[1]


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_document(content: bytes, filename: str, media_type: str) -> list[LoadedPage]:
    kind = validate_document_type(filename, media_type)
    if kind == "pdf":
        try:
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
```

- [ ] **Step 4: Write and implement safe file-store tests**

```python
# tests/storage/test_files.py
from pathlib import Path

from app.storage.files import LocalDocumentStore


def test_store_ignores_user_path_components(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)
    saved = store.save("doc123", "../../secret.pdf", b"pdf")
    assert saved == (tmp_path / "doc123.pdf").resolve()
    assert saved.read_bytes() == b"pdf"


def test_delete_is_confined_to_upload_root(tmp_path: Path) -> None:
    store = LocalDocumentStore(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    assert store.delete(outside) is False
    assert outside.exists()
```

`LocalDocumentStore.save()` must derive the stored filename from the generated document ID and validated suffix; it must verify `target.resolve().is_relative_to(root.resolve())` before writing. `delete()` returns `False` for paths outside the root.

- [ ] **Step 5: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/retrieval/test_loaders.py tests/storage/test_files.py -q`

Expected: PASS.

- [ ] **Step 6: Commit loaders and local storage**

```powershell
git add app/retrieval app/storage/files.py tests/retrieval/test_loaders.py tests/storage/test_files.py
git commit -m "feat: add safe local document loading"
```

---

### Task 2: Deterministic Page-Local Chunking

**Files:**
- Create: `app/retrieval/chunking.py`
- Create: `tests/retrieval/test_chunking.py`

**Interfaces:**
- Consumes: `LoadedPage`, `EvidenceChunk`, document ID and filename.
- Produces: `make_evidence_id()`, `chunk_pages()`.

- [ ] **Step 1: Write failing chunking tests**

```python
# tests/retrieval/test_chunking.py
from app.domain.documents import LoadedPage
from app.retrieval.chunking import chunk_pages


def test_chunks_do_not_cross_page_boundaries() -> None:
    chunks = chunk_pages(
        document_id="doc1",
        filename="paper.pdf",
        pages=[LoadedPage(page_number=1, text="A" * 1500), LoadedPage(page_number=2, text="B" * 200)],
        chunk_size=1200,
        overlap=200,
    )
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(set(chunk.content) <= ({"A"} if chunk.page_number == 1 else {"B"}) for chunk in chunks)


def test_chunk_ids_are_stable() -> None:
    kwargs = dict(
        document_id="doc1", filename="a.md",
        pages=[LoadedPage(page_number=None, text="alpha\n\nbeta")],
        chunk_size=20, overlap=5,
    )
    assert [c.id for c in chunk_pages(**kwargs)] == [c.id for c in chunk_pages(**kwargs)]


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages("d", "a.txt", [LoadedPage(page_number=None, text="text")], 100, 100)
```

Add tests for global zero-based `chunk_index`, paragraph-preferred boundaries, no blank chunks, and changed content producing a changed ID.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/retrieval/test_chunking.py -v`

Expected: FAIL because `chunk_pages` does not exist.

- [ ] **Step 3: Implement deterministic splitting**

```python
# app/retrieval/chunking.py
from hashlib import sha256

from app.domain.documents import EvidenceChunk, LoadedPage

SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", " ")


def make_evidence_id(document_id: str, page_number: int | None, chunk_index: int, digest: str) -> str:
    raw = f"{document_id}:{page_number or 0}:{chunk_index}:{digest}".encode("utf-8")
    return "ev_" + sha256(raw).hexdigest()


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
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


def chunk_pages(document_id: str, filename: str, pages: list[LoadedPage], chunk_size: int, overlap: int) -> list[EvidenceChunk]:
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    result: list[EvidenceChunk] = []
    for page in pages:
        for text in _split_text(page.text, chunk_size, overlap):
            digest = sha256(text.encode("utf-8")).hexdigest()
            index = len(result)
            result.append(EvidenceChunk(
                id=make_evidence_id(document_id, page.page_number, index, digest),
                document_id=document_id,
                filename=filename,
                page_number=page.page_number,
                chunk_index=index,
                content=text,
                content_sha256=digest,
            ))
    return result
```

- [ ] **Step 4: Run chunking tests**

Run: `.venv\Scripts\python.exe -m pytest tests/retrieval/test_chunking.py -v`

Expected: PASS.

- [ ] **Step 5: Commit deterministic chunking**

```powershell
git add app/retrieval/chunking.py tests/retrieval/test_chunking.py
git commit -m "feat: add traceable document chunking"
```

---

### Task 3: Embedding Pipeline and Document Service

**Files:**
- Create: `app/retrieval/embedding.py`
- Create: `app/services/__init__.py`
- Create: `app/services/documents.py`
- Create: `tests/retrieval/test_embedding.py`
- Create: `tests/services/test_document_service.py`

**Interfaces:**
- Consumes: loaders, chunker, `EmbeddingProvider`, `DocumentRepository`, `LocalDocumentStore`, `Settings`.
- Produces: `embed_chunks() -> tuple[list[tuple[EvidenceChunk, list[float]]], list[ProviderMetadata]]`, `DocumentService.ingest()`, `list_documents()`, `delete_document()`.

- [ ] **Step 1: Write failing embedding tests**

```python
# tests/retrieval/test_embedding.py
from app.providers.fake import DeterministicEmbeddingProvider
from app.retrieval.embedding import embed_chunks


def test_embed_chunks_preserves_order_and_model(chunks) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=16)
    embedded, metrics = embed_chunks(chunks, provider, batch_size=2)
    assert [item[0].id for item in embedded] == [chunk.id for chunk in chunks]
    assert all(item[0].embedding_model == "fake-hash-16" for item in embedded)
    assert all(len(item[1]) == 16 for item in embedded)
    assert metrics
    assert all(item.model == "fake-hash-16" for item in metrics)


def test_embed_chunks_rejects_count_mismatch(chunks) -> None:
    provider = BrokenEmbeddingProvider(return_count=1)
    with pytest.raises(RetrievalError, match="embedding_count_mismatch"):
        embed_chunks(chunks, provider, batch_size=32)
```

Add a 65-chunk test asserting batch sizes `32, 32, 1`, a dimension mismatch test, and normalization assertions.

- [ ] **Step 2: Implement batched embedding validation**

```python
# app/retrieval/embedding.py
from dataclasses import replace
from math import isfinite, sqrt

from app.domain.errors import RetrievalError


def _normalize(vector: list[float]) -> list[float]:
    if not vector or not all(isfinite(value) for value in vector):
        raise RetrievalError("invalid_embedding", "embedding must contain finite values")
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise RetrievalError("zero_embedding", "embedding vector cannot be zero")
    return [value / norm for value in vector]


def embed_chunks(chunks, provider, batch_size: int):
    embedded = []
    metrics = []
    dimensions = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors, metadata = provider.embed_documents([chunk.content for chunk in batch])
        metrics.append(metadata)
        if len(vectors) != len(batch):
            raise RetrievalError("embedding_count_mismatch", "provider returned the wrong number of vectors")
        for chunk, vector in zip(batch, vectors):
            normalized = _normalize(vector)
            dimensions = dimensions or len(normalized)
            if len(normalized) != dimensions:
                raise RetrievalError("embedding_dimension_mismatch", "embedding dimensions changed within one document")
            embedded.append((chunk.model_copy(update={"embedding_model": provider.model_name}), normalized))
    return embedded, metrics
```

- [ ] **Step 3: Write document-service tests**

```python
# tests/services/test_document_service.py
def test_ingest_persists_ready_document_chunks_and_file(service, repository) -> None:
    document = service.ingest("notes.md", "text/markdown", b"Alpha\n\nBeta")
    assert document.status == "ready"
    assert repository.get_document(document.id) == document
    assert repository.list_chunks([document.id])


def test_identical_content_is_idempotent(service, embedding_provider) -> None:
    first = service.ingest("first.md", "text/markdown", b"same")
    second = service.ingest("second.md", "text/markdown", b"same")
    assert second.id == first.id
    assert embedding_provider.document_calls == 1


def test_provider_failure_marks_document_failed_without_chunks(service, repository) -> None:
    service.embedding_provider = FailingEmbeddingProvider()
    with pytest.raises(RetrievalError):
        service.ingest("notes.txt", "text/plain", b"text")
    failed = repository.list_documents()[0]
    assert failed.status == "failed"
    assert repository.list_chunks([failed.id]) == []
```

Also test upload size before disk write, empty files, deletion of data and stored bytes, and unknown document IDs.

- [ ] **Step 4: Implement `DocumentService` transaction order**

```python
class DocumentService:
    def ingest(self, filename: str, media_type: str, content: bytes) -> DocumentRecord:
        if len(content) > self.settings.max_upload_file_bytes:
            raise DocumentError("document_too_large", "document exceeds configured size limit")
        validate_document_type(filename, media_type)
        digest = sha256(content).hexdigest()
        existing = self.repository.get_document_by_sha256(digest)
        if existing and existing.status == "ready":
            return existing
        if existing and existing.status == "processing":
            raise DocumentError("document_ingestion_in_progress", "identical document is already being processed")
        document = (
            existing.model_copy(update={"filename": filename, "media_type": media_type, "status": "processing", "error_message": None})
            if existing
            else DocumentRecord.new(filename, media_type, digest)
        )
        stored_path = self.file_store.save(document.id, filename, content)
        document = document.model_copy(update={"storage_path": str(stored_path)})
        if existing:
            self.repository.update_document(document)
        else:
            self.repository.add_document(document)
        try:
            pages = load_document(content, filename, media_type)
            chunks = chunk_pages(document.id, filename, pages, self.settings.chunk_size, self.settings.chunk_overlap)
            indexed, _embedding_metrics = embed_chunks(
                chunks, self.embedding_provider, batch_size=self.settings.embedding_batch_size
            )
            document = document.model_copy(update={"status": "ready", "page_count": len(pages)})
            self.repository.complete_ingestion(document, indexed)
            return document
        except Exception as exc:
            self.repository.fail_ingestion(document.id, type(exc).__name__)
            raise
```

`complete_ingestion()` and `fail_ingestion()` are the exact atomic repository operations from plan 1. The embedding call still returns safe metadata for tracing even though document ingestion does not attach it to a research run. Do not delete the failed upload automatically; retain it for explicit user cleanup but never retain partial chunks.

- [ ] **Step 5: Run service and retrieval tests**

Run: `.venv\Scripts\python.exe -m pytest tests/retrieval tests/services/test_document_service.py -q`

Expected: PASS.

- [ ] **Step 6: Commit ingestion service**

```powershell
git add app/retrieval/embedding.py app/services tests/retrieval/test_embedding.py tests/services/test_document_service.py
git commit -m "feat: persist embedded documents"
```

---

### Task 4: Local Cosine Vector Index

**Files:**
- Create: `app/retrieval/index.py`
- Create: `tests/retrieval/test_index.py`

**Interfaces:**
- Consumes: `DocumentRepository.list_chunks()`.
- Produces: `LocalVectorIndex.search()`.

- [ ] **Step 1: Write failing ranking tests**

```python
# tests/retrieval/test_index.py
def test_search_orders_by_cosine_similarity(repository) -> None:
    repository.seed_vectors({"a": [1.0, 0.0], "b": [0.8, 0.2], "c": [0.0, 1.0]})
    index = LocalVectorIndex(repository)
    results = index.search([1.0, 0.0], ["doc1"], limit=2, min_score=0.15)
    assert [item.id for item in results] == ["a", "b"]
    assert results[0].score == pytest.approx(1.0)


def test_search_filters_documents_and_minimum_score(repository) -> None:
    repository.seed_vectors({"a": [1.0, 0.0], "c": [0.0, 1.0]}, document_ids={"a": "doc1", "c": "doc2"})
    results = LocalVectorIndex(repository).search([1.0, 0.0], ["doc1"], 10, 0.2)
    assert [item.id for item in results] == ["a"]
```

Add zero/NaN/Inf query and candidate-vector tests, dimension-mismatch, stable tie-break by Evidence ID, and empty-document-set tests.

- [ ] **Step 2: Run tests and observe failure**

Run: `.venv\Scripts\python.exe -m pytest tests/retrieval/test_index.py -v`

- [ ] **Step 3: Implement normalized cosine search**

```python
# app/retrieval/index.py
import numpy as np

from app.domain.errors import RetrievalError


class LocalVectorIndex:
    def __init__(self, repository) -> None:
        self.repository = repository

    def search(self, query_vector, document_ids, limit: int, min_score: float):
        rows = self.repository.list_chunks(list(document_ids))
        if not rows:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if query.ndim != 1 or not np.all(np.isfinite(query)) or not np.isfinite(norm) or norm == 0:
            raise RetrievalError("invalid_query_embedding", "query embedding must be a non-zero vector")
        query = query / norm
        ranked = []
        for chunk, vector in rows:
            candidate = np.asarray(vector, dtype=np.float32)
            if candidate.shape != query.shape:
                raise RetrievalError("embedding_dimension_mismatch", "query and document embedding dimensions differ")
            candidate_norm = np.linalg.norm(candidate)
            if not np.all(np.isfinite(candidate)) or not np.isfinite(candidate_norm) or candidate_norm == 0:
                raise RetrievalError("invalid_stored_embedding", f"stored embedding {chunk.id} is not a finite non-zero vector")
            score = float(np.dot(query, candidate / candidate_norm))
            if score >= min_score:
                ranked.append(chunk.model_copy(update={"score": score}))
        return sorted(ranked, key=lambda item: (-float(item.score), item.id))[:limit]
```

- [ ] **Step 4: Run index tests**

Run: `.venv\Scripts\python.exe -m pytest tests/retrieval/test_index.py -v`

Expected: PASS.

- [ ] **Step 5: Commit local vector search**

```powershell
git add app/retrieval/index.py tests/retrieval/test_index.py
git commit -m "feat: add local vector search"
```

---

### Task 5: Query Deduplication, Reciprocal-Rank Fusion, and Retriever Service

**Files:**
- Create: `app/retrieval/fusion.py`
- Create: `app/retrieval/retriever.py`
- Create: `tests/retrieval/test_fusion.py`
- Create: `tests/retrieval/test_retriever.py`

**Interfaces:**
- Consumes: `EmbeddingProvider`, `LocalVectorIndex`, settings.
- Produces: `normalize_queries()`, `reciprocal_rank_fuse()`, `RetrievalBatch`, `EvidenceRetriever.retrieve()`.

- [ ] **Step 1: Write failing fusion tests**

```python
# tests/retrieval/test_fusion.py
def test_normalize_queries_keeps_original_first_and_deduplicates() -> None:
    assert normalize_queries(" Main question ", ["query", "QUERY", "extra"], max_expansions=2) == [
        "Main question", "query", "extra"
    ]


def test_normalize_queries_with_zero_expansions_keeps_only_question() -> None:
    assert normalize_queries("Question", ["extra"], max_expansions=0) == ["Question"]


def test_normalize_queries_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="question"):
        normalize_queries("   ", ["extra"], max_expansions=1)


def test_rrf_rewards_evidence_seen_in_multiple_rankings() -> None:
    result = reciprocal_rank_fuse([[chunk("a"), chunk("b")], [chunk("b"), chunk("c")]], rrf_k=60, top_k=3)
    assert [item.id for item in result] == ["b", "a", "c"]
```

Add tests for empty lists, duplicate IDs inside one ranking, `top_k`, and stable ID tie-breaks.

- [ ] **Step 2: Implement fusion helpers**

```python
# app/retrieval/fusion.py
def normalize_queries(question: str, expansions: list[str], max_expansions: int) -> list[str]:
    if max_expansions < 0:
        raise ValueError("max_expansions must be non-negative")
    original = question.strip()
    if not original:
        raise ValueError("question must not be blank")
    result = [original]
    seen = {original.casefold()}
    accepted_expansions = 0
    for value in expansions:
        if accepted_expansions >= max_expansions:
            break
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
            accepted_expansions += 1
    return result


def reciprocal_rank_fuse(ranked_lists, rrf_k: int, top_k: int):
    scores: dict[str, float] = {}
    evidence = {}
    for ranked in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            if item.id in seen_in_list:
                continue
            seen_in_list.add(item.id)
            evidence[item.id] = item
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda evidence_id: (-scores[evidence_id], evidence_id))
    return [evidence[evidence_id].model_copy(update={"score": scores[evidence_id]}) for evidence_id in ordered[:top_k]]
```

- [ ] **Step 3: Write Retriever integration tests**

```python
# tests/retrieval/test_retriever.py
def test_retriever_embeds_each_unique_query_once(fake_embeddings, fake_index) -> None:
    retriever = EvidenceRetriever(
        fake_embeddings, fake_index, candidate_k=20, rrf_k=60,
        min_similarity=0.15, max_query_expansions=4,
    )
    batch = retriever.retrieve("Question", ["query", "QUERY"], ["doc1"], top_k=6)
    assert fake_embeddings.queries == ["Question", "query"]
    assert len(batch.evidence) <= 6
    assert len(batch.provider_metrics) == 2
```

Also test no document IDs, no candidates, embedding errors, and propagation of filename/page/chunk metadata.

- [ ] **Step 4: Implement the Retriever**

```python
# app/retrieval/retriever.py
from app.retrieval.contracts import RetrievalBatch


class EvidenceRetriever:
    def __init__(self, embedding_provider, index, candidate_k: int = 20, rrf_k: int = 60, min_similarity: float = 0.15, max_query_expansions: int = 4) -> None:
        self.embedding_provider = embedding_provider
        self.index = index
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.min_similarity = min_similarity
        self.max_query_expansions = max_query_expansions

    def retrieve(self, question: str, expansions: list[str], document_ids: list[str], top_k: int):
        if not document_ids:
            raise RetrievalError("no_documents_selected", "at least one document is required")
        queries = normalize_queries(question, expansions, self.max_query_expansions)
        ranked = []
        provider_metrics = []
        for query in queries:
            vector, metadata = self.embedding_provider.embed_query(query)
            provider_metrics.append(metadata)
            ranked.append(
                self.index.search(
                    vector,
                    document_ids,
                    self.candidate_k,
                    min_score=self.min_similarity,
                )
            )
        return RetrievalBatch(
            evidence=reciprocal_rank_fuse(ranked, self.rrf_k, top_k),
            provider_metrics=provider_metrics,
        )
```

Add the upstream header referencing `EU_Green_Compliance_FAQ_Bot.ipynb` cells 21, 25, and 36, noting that source metadata, score semantics, and duplicate handling were corrected.

- [ ] **Step 5: Run the phase gate**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/retrieval tests/storage/test_files.py tests/services/test_document_service.py -q
```

Expected: PASS without network or API keys.

- [ ] **Step 6: Commit fused retrieval**

```powershell
git add app/retrieval/fusion.py app/retrieval/retriever.py tests/retrieval/test_fusion.py tests/retrieval/test_retriever.py
git commit -m "feat: add fused evidence retrieval"
```
