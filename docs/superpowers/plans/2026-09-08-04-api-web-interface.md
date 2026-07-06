# FastAPI and Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the document and research services through safe REST/SSE endpoints and deliver a self-contained browser interface for uploading sources, following Agent progress, and reading citation-linked reports.

**Architecture:** A composition root builds providers, repositories, retrieval, Agents, graph, and application services. FastAPI routes depend only on service interfaces. A small thread-pool task manager executes synchronous research workflows while async SSE handlers stream persisted events to the browser.

**Tech Stack:** FastAPI 0.112.2, Starlette, Uvicorn 0.30.6, Pydantic 2.8.2, vanilla HTML/CSS/JavaScript, pytest 8.3.2.

**Spec:** `docs/superpowers/specs/2026-09-08-multi-agent-research-assistant-design.md`

## Global Constraints

- Serve UI and API from the same origin; do not add Node, npm, React, Vite, CDN scripts, or external fonts.
- API routes do not import Agent implementations, provider SDKs, or SQL.
- Every error response contains `code`, `message`, and `request_id`; stack traces are never returned.
- Every response returns a safe `X-Request-ID`.
- Upload limits are enforced while streaming bytes, not by trusting `Content-Length`.
- SSE supports reconnect cursors and closes after all terminal events are delivered.
- Model output is inserted with `textContent` or explicit DOM nodes, never `innerHTML`.
- The browser never receives or stores an API key.
- Default API tests use fake application services and perform no network requests.

---

### Task 1: Composition Root, Application Factory, and Health Endpoint

**Files:**
- Create: `app/bootstrap.py`
- Create: `app/api/__init__.py`
- Create: `app/api/dependencies.py`
- Create: `app/api/schemas.py`
- Create: `app/api/middleware.py`
- Create: `app/api/errors.py`
- Create: `app/observability.py`
- Create: `app/api/routes/__init__.py`
- Create: `app/api/routes/health.py`
- Create: `app/main.py`
- Create: `tests/api/conftest.py`
- Create: `tests/api/test_health.py`
- Create: `tests/api/test_errors.py`
- Create: `tests/api/test_logging.py`

**Interfaces:**
- Consumes: `Settings`, repositories, providers, retrieval, graph, `DocumentService`, `ResearchService`.
- Produces: `ApplicationContainer`, `ApplicationServices`, `build_container()`, `create_app()`, and structured safe logging helpers.

- [ ] **Step 1: Write failing application-boundary tests**

```python
# tests/api/test_health.py
def test_health_reports_only_safe_readiness(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "1.0.0",
        "database_ready": True,
        "providers": {"chat": True, "embedding": True},
    }
    assert "api_key" not in response.text.casefold()


def test_request_id_is_echoed_when_safe(client) -> None:
    response = client.get("/health", headers={"X-Request-ID": "demo-123"})
    assert response.headers["X-Request-ID"] == "demo-123"
```

```python
# tests/api/test_errors.py
def test_unknown_route_uses_stable_error_shape(client) -> None:
    response = client.get("/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
```

- [ ] **Step 2: Run tests and observe failure**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_health.py tests/api/test_errors.py -v`

Expected: FAIL because `app.main` and API modules do not exist.

- [ ] **Step 3: Define API schemas and service container**

```python
# app/api/schemas.py
from typing import Any, Literal
from pydantic import BaseModel, Field


class ApiErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ApiErrorResponse(BaseModel):
    error: ApiErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database_ready: bool
    providers: dict[str, bool]
```

```python
# app/api/dependencies.py
from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True)
class ApplicationServices:
    documents: object
    research: object


def get_services(request: Request) -> ApplicationServices:
    return request.app.state.services
```

- [ ] **Step 4: Implement composition and app lifespan**

```python
# app/bootstrap.py
from dataclasses import dataclass


@dataclass
class ApplicationContainer:
    settings: Settings
    database: Database
    chat_provider: object
    embedding_provider: object
    services: ApplicationServices


def build_container(settings: Settings | None = None) -> ApplicationContainer:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    database.initialize()
    documents = DocumentRepository(database)
    runs = RunRepository(database)
    chat = build_chat_provider(settings)
    embeddings = build_embedding_provider(settings)
    document_service = DocumentService(documents, LocalDocumentStore(settings.upload_dir), embeddings, settings)
    retriever = EvidenceRetriever(
        embeddings,
        LocalVectorIndex(documents),
        candidate_k=settings.retrieval_candidate_k,
        rrf_k=settings.retrieval_rrf_k,
        min_similarity=settings.retrieval_min_similarity,
        max_query_expansions=settings.max_query_expansions,
    )
    dependencies = WorkflowDependencies(
        planner=PlannerAgent(chat),
        retriever=RetrieverAgent(retriever, top_k=settings.retrieval_top_k),
        researcher=ResearcherAgent(chat),
        critic=CriticAgent(chat),
        writer=WriterAgent(chat),
        citation_validator=CitationValidatorNode(),
        max_iterations=settings.max_revision_iterations,
        node_wrapper=lambda stage, handler: checkpointed_node(stage, handler, runs),
    )
    graph = build_research_graph(dependencies)
    provider_name, model_name = configured_chat_identity(settings)
    research_service = ResearchService(
        documents, runs, graph, provider_name, model_name,
        recursion_limit=settings.max_workflow_steps,
    )
    services = ApplicationServices(documents=document_service, research=research_service)
    return ApplicationContainer(settings, database, chat, embeddings, services)
```

Refine model selection in the actual code so OpenAI and fake settings report the correct configured model. If chat and embedding factories return the same closeable object, close it once.

`configured_chat_identity()` returns the selected provider/model pair for DashScope, OpenAI, or fake mode. `create_app(container=None, services=None)` uses an async lifespan to create or accept the container and register routers. Task 3 extends the lifespan with `RunTaskManager`, startup recovery, and deterministic provider shutdown.

- [ ] **Step 5: Implement request ID, security headers, and error handlers**

Accepted request IDs match `[A-Za-z0-9._-]{1,128}`; invalid or absent values are replaced with `uuid4().hex`.

Set:

```text
X-Content-Type-Options: nosniff
Referrer-Policy: no-referrer
Content-Security-Policy: default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'
```

Map `DocumentError`, `RetrievalError`, `WorkflowError`, `ProviderError`, request validation, HTTP 404, and unexpected errors to stable JSON. Unexpected errors return `internal_error` and `Internal server error` only.

Implement `app/observability.py` with JSON logging helpers whose stable fields are `request_id`, `run_id`, `stage`, `provider`, `model`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `retries`, and `error_type`. A redaction filter removes values for keys matching `api_key`, `authorization`, `token_value`, or `secret`; tests with `caplog` assert a sentinel key never appears while safe metadata does.

- [ ] **Step 6: Run tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_health.py tests/api/test_errors.py tests/api/test_logging.py -q`

Expected: PASS.

```powershell
git add app/bootstrap.py app/api app/observability.py app/main.py tests/api/conftest.py tests/api/test_health.py tests/api/test_errors.py tests/api/test_logging.py
git commit -m "feat: add FastAPI application boundary"
```

---

### Task 2: Document Upload and Listing API

**Files:**
- Modify: `app/api/schemas.py`
- Create: `app/api/routes/documents.py`
- Create: `tests/api/test_documents.py`

**Interfaces:**
- Consumes: `DocumentService`.
- Produces: `POST /api/documents`, `GET /api/documents`, `DELETE /api/documents/{document_id}`.

- [ ] **Step 1: Write failing endpoint tests**

```python
# tests/api/test_documents.py
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


def test_upload_rejects_stream_over_limit(client) -> None:
    response = client.post(
        "/api/documents",
        files=[("files", ("large.txt", b"x" * 1025, "text/plain"))],
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_upload_rejects_combined_request_over_limit(client) -> None:
    response = client.post(
        "/api/documents",
        files=[
            ("files", ("a.txt", b"a" * 700, "text/plain")),
            ("files", ("b.txt", b"b" * 700, "text/plain")),
        ],
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_total_too_large"


def test_list_and_delete_documents(client) -> None:
    assert client.get("/api/documents").status_code == 200
    assert client.delete("/api/documents/doc1").status_code == 204
```

- [ ] **Step 2: Run tests and observe missing route**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_documents.py -v`

Expected: FAIL with 404 responses.

- [ ] **Step 3: Add response schemas**

```python
class DocumentResponse(BaseModel):
    id: str
    filename: str
    media_type: str
    sha256: str
    status: str
    page_count: int
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    documents: list[DocumentResponse]
```

- [ ] **Step 4: Implement bounded upload reading**

```python
async def read_upload_limited(upload: UploadFile, per_file_limit: int, remaining: int) -> bytes:
    data = bytearray()
    try:
        while chunk := await upload.read(64 * 1024):
            data.extend(chunk)
            if len(data) > per_file_limit or len(data) > remaining:
                raise HTTPException(status_code=413, detail={"code": "upload_too_large"})
        return bytes(data)
    finally:
        await upload.close()
```

The route passes `settings.max_upload_file_bytes` as `per_file_limit`, tracks bytes already accepted, and rejects the request once the cumulative count exceeds `settings.max_upload_total_bytes`. It first reads and validates every upload, then calls `DocumentService.ingest()` through `run_in_threadpool`. This prevents partial ingestion when a later file exceeds the request limit.

- [ ] **Step 5: Implement routes**

```python
@router.post("", status_code=201, response_model=DocumentUploadResponse)
async def upload_documents(files: list[UploadFile], request: Request, services=Depends(get_services)): ...

@router.get("", response_model=list[DocumentResponse])
def list_documents(services=Depends(get_services)): ...

@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, services=Depends(get_services)): ...
```

Reject an empty filename, empty file list, unsupported MIME pair, per-file overflow, and total-request overflow with stable error codes.

- [ ] **Step 6: Run API tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_documents.py -q`

Expected: PASS.

```powershell
git add app/api/schemas.py app/api/routes/documents.py tests/api/test_documents.py
git commit -m "feat: expose document management API"
```

---

### Task 3: Background Research Execution and REST Status

**Files:**
- Create: `app/api/tasks.py`
- Modify: `app/api/schemas.py`
- Modify: `app/api/dependencies.py`
- Create: `app/api/routes/research.py`
- Modify: `app/main.py`
- Create: `tests/api/test_tasks.py`
- Create: `tests/api/test_research.py`

**Interfaces:**
- Consumes: synchronous `ResearchService`.
- Produces: `RunTaskManager`, `POST /api/research`, `GET /api/research/{run_id}`.

- [ ] **Step 1: Write failing task-manager tests**

```python
# tests/api/test_tasks.py
def test_task_manager_deduplicates_active_run(fake_research_service) -> None:
    manager = RunTaskManager(fake_research_service, max_workers=1)
    manager.start("run1")
    manager.start("run1")
    fake_research_service.release.set()
    manager.wait("run1", timeout=2)
    assert fake_research_service.execution_count == 1
    manager.close()
```

Add tests that `close()` waits for running work and cancels only queued futures,
and that one failed run is persisted as failed while a subsequently submitted
run still completes in the same manager.

- [ ] **Step 2: Implement bounded thread-pool execution**

```python
# app/api/tasks.py
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class RunTaskManager:
    def __init__(self, research_service, max_workers: int = 2) -> None:
        self.service = research_service
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research")
        self.futures: dict[str, Future] = {}
        self.lock = Lock()

    def start(self, run_id: str) -> None:
        with self.lock:
            current = self.futures.get(run_id)
            if current is not None and not current.done():
                return
            future = self.executor.submit(self.service.execute, run_id)
            self.futures[run_id] = future
        future.add_done_callback(lambda completed: self._discard(run_id, completed))

    def _discard(self, run_id: str, completed: Future) -> None:
        with self.lock:
            if self.futures.get(run_id) is completed:
                self.futures.pop(run_id, None)
        if completed.cancelled():
            return
        exception = completed.exception()
        if exception is not None:
            logger.error(
                "research task failed",
                extra={"run_id": run_id, "error_type": type(exception).__name__},
            )

    def wait(self, run_id: str, timeout: float | None = None):
        with self.lock:
            future = self.futures.get(run_id)
        return None if future is None else future.result(timeout=timeout)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
```

The callback is attached after releasing the lock because `add_done_callback()` may execute immediately for an already-completed future. It logs only run ID and exception type; workflow failure persistence remains the service's responsibility.

- [ ] **Step 3: Write failing REST tests**

```python
# tests/api/test_research.py
def test_create_research_returns_202_and_location(client, task_manager) -> None:
    response = client.post(
        "/api/research",
        json={"question": "What does the evidence show?", "document_ids": ["doc1"]},
    )
    assert response.status_code == 202
    assert response.headers["Location"] == "/api/research/run1"
    assert response.json() == {"run_id": "run1", "status": "queued"}
    assert task_manager.started == ["run1"]


def test_get_completed_run_includes_report(client) -> None:
    response = client.get("/api/research/run1")
    assert response.status_code == 200
    assert response.json()["report"]["citations"][0]["evidence_id"] == "ev_good"
```

Add blank question, empty documents, duplicate document IDs preserving first appearance, unknown run, failed run, and insufficient-evidence response tests.

- [ ] **Step 4: Add request and response schemas**

```python
class ResearchCreateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    document_ids: list[str] = Field(min_length=1, max_length=100)


class ResearchCreateResponse(BaseModel):
    run_id: str
    status: str


class ResearchRunResponse(BaseModel):
    run: ResearchRun
    report: ResearchReport | None
```

`ResearchRun.evidence_sufficient` is `None` before completion, `True` for an
ordinary grounded result, and `False` when the Critic reaches its revision
limit. The latter is still a successful `completed` run, but the response and
browser must label it as “证据不足” and retain the report limitations.

- [ ] **Step 5: Implement research routes**

```python
@router.post("", status_code=202, response_model=ResearchCreateResponse)
def create_research(payload: ResearchCreateRequest, response: Response, services=Depends(get_services), tasks=Depends(get_task_manager)):
    document_ids = list(dict.fromkeys(payload.document_ids))
    run = services.research.create_run(payload.question, document_ids)
    tasks.start(run.id)
    response.headers["Location"] = f"/api/research/{run.id}"
    return ResearchCreateResponse(run_id=run.id, status=run.status)


@router.get("/{run_id}", response_model=ResearchRunResponse)
def get_research(run_id: str, services=Depends(get_services)):
    run = services.research.get_run(run_id)
    return ResearchRunResponse(run=run, report=services.research.get_report(run_id))
```

- [ ] **Step 6: Wire startup recovery**

At application startup:

```python
for run_id in services.research.list_incomplete_ids():
    task_manager.start(run_id)
```

Add `ResearchService.list_incomplete_ids() -> list[str]` as a read-only repository query returning only `queued` and `running` rows; do not execute recovery synchronously during startup. In `create_app`, assign the manager to `app.state.run_tasks`, expose it through `get_task_manager`, schedule recovery in lifespan startup, and call `close()` during shutdown before closing provider clients.

- [ ] **Step 7: Run tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_tasks.py tests/api/test_research.py -q`

Expected: PASS.

```powershell
git add app/api/tasks.py app/api/schemas.py app/api/routes/research.py app/main.py tests/api/test_tasks.py tests/api/test_research.py
git commit -m "feat: run research tasks through the API"
```

---

### Task 4: Resumable Server-Sent Events

**Files:**
- Create: `app/api/sse.py`
- Modify: `app/api/routes/research.py`
- Create: `tests/api/test_sse.py`

**Interfaces:**
- Consumes: `ResearchService.list_events()`, `get_run()`.
- Produces: `encode_sse()`, `stream_run_events()`, `GET /api/research/{run_id}/events`.

- [ ] **Step 1: Write failing frame and resume tests**

```python
# tests/api/test_sse.py
def test_encode_sse_has_id_event_and_compact_json(event) -> None:
    frame = encode_sse(event).decode("utf-8")
    assert frame.startswith("id: 7\nevent: workflow\n")
    assert 'data: {"sequence":7' in frame
    assert frame.endswith("\n\n")


def test_event_stream_resumes_after_cursor(client) -> None:
    with client.stream(
        "GET", "/api/research/run1/events?after=2", headers={"Last-Event-ID": "3"}
    ) as response:
        payload = "".join(response.iter_text())
    assert "id: 3\n" not in payload
    assert "id: 4\n" in payload
```

- [ ] **Step 2: Implement exact SSE encoding**

```python
# app/api/sse.py
def encode_sse(event: RunEvent) -> bytes:
    data = event.model_dump_json(exclude_none=True)
    return f"id: {event.sequence}\nevent: workflow\ndata: {data}\n\n".encode("utf-8")
```

- [ ] **Step 3: Implement event generator**

```python
async def stream_run_events(request, run_id, service, after_sequence, poll_interval, heartbeat_interval):
    cursor = after_sequence
    last_sent = monotonic()
    while not await request.is_disconnected():
        events = await run_in_threadpool(service.list_events, run_id, cursor)
        for event in events:
            cursor = event.sequence
            last_sent = monotonic()
            yield encode_sse(event)
        run = await run_in_threadpool(service.get_run, run_id)
        if run.status in {"completed", "failed"} and not events:
            break
        if monotonic() - last_sent >= heartbeat_interval:
            last_sent = monotonic()
            yield b": keep-alive\n\n"
        await asyncio.sleep(poll_interval)
```

Use the larger of query `after` and valid integer `Last-Event-ID`. Validate the run before returning `StreamingResponse`. Set `Cache-Control: no-cache` and `X-Accel-Buffering: no`.

- [ ] **Step 4: Run SSE tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_sse.py -q`

Expected: reconnect, heartbeat, disconnect, terminal completion, and missing-run cases PASS.

```powershell
git add app/api/sse.py app/api/routes/research.py tests/api/test_sse.py
git commit -m "feat: stream research workflow events"
```

---

### Task 5: Self-Contained Browser Dashboard

**Files:**
- Create: `app/static/index.html`
- Create: `app/static/styles.css`
- Create: `app/static/app.js`
- Create: `app/api/routes/ui.py`
- Create: `tests/static/test_static_ui.py`

**Interfaces:**
- Consumes: document REST API, research REST API, SSE endpoint.
- Produces: `GET /`, `/static/styles.css`, `/static/app.js`.

- [ ] **Step 1: Write failing static-surface tests**

```python
# tests/static/test_static_ui.py
from pathlib import Path


def test_index_contains_complete_research_controls(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    for element_id in (
        "upload-form", "document-input", "document-list", "research-form",
        "question-input", "research-documents", "timeline", "report", "citation-dialog",
    ):
        assert f'id="{element_id}"' in response.text


def test_frontend_is_self_contained_and_avoids_html_injection() -> None:
    html = Path("app/static/index.html").read_text(encoding="utf-8")
    script = Path("app/static/app.js").read_text(encoding="utf-8")
    assert "https://" not in html and "http://" not in html
    assert "innerHTML" not in script
    assert "EventSource" in script
```

- [ ] **Step 2: Run tests and observe missing static files**

Run: `.venv\Scripts\python.exe -m pytest tests/static/test_static_ui.py -v`

- [ ] **Step 3: Create semantic HTML and responsive CSS**

`index.html` contains:

```html
<main class="app-shell">
  <section aria-labelledby="documents-heading">
    <form id="upload-form"><input id="document-input" name="files" type="file" multiple accept=".pdf,.md,.markdown,.txt"></form>
    <div id="document-list" aria-live="polite"></div>
  </section>
  <section aria-labelledby="research-heading">
    <form id="research-form"><textarea id="question-input" required minlength="3"></textarea><div id="research-documents"></div></form>
    <ol id="timeline" aria-live="polite"></ol>
    <article id="report" aria-live="polite"></article>
  </section>
  <dialog id="citation-dialog"><button id="citation-close" type="button">关闭</button><div id="citation-content"></div></dialog>
</main>
```

CSS uses system fonts, visible focus rings, WCAG-readable contrast, cards, stage status chips, and a `@media (max-width: 760px)` single-column layout.

- [ ] **Step 4: Implement safe JavaScript behavior**

```javascript
async function apiFetch(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ error: { message: response.statusText } }));
    throw new Error(body.error?.message || "请求失败");
  }
  return response.status === 204 ? null : response.json();
}

function appendText(parent, tagName, text, className = "") {
  const node = document.createElement(tagName);
  node.textContent = text;
  if (className) node.className = className;
  parent.appendChild(node);
  return node;
}
```

Implement `refreshDocuments`, `uploadDocuments`, `startResearch`, `openEventStream`, `renderWorkflowEvent`, `renderReport`, and `openCitation` only with DOM APIs and `textContent`. Citation buttons use an Evidence ID stored in `dataset.evidenceId`; they retrieve details from the already returned `report.citations` array.

When a completed response has `run.evidence_sufficient === false`, render a visible “证据不足” status before the report and list its limitations. This is a successful terminal run, not an HTTP error; the final SSE `finished` event carries the same boolean.

- [ ] **Step 5: Run static and API regression tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/static tests/api -q
node --check app/static/app.js
```

Expected: PASS and JavaScript syntax exit code 0.

- [ ] **Step 6: Commit the dashboard**

```powershell
git add app/static app/api/routes/ui.py app/main.py tests/static
git commit -m "feat: add browser research dashboard"
```

---

### Task 6: Full Local Product Smoke Test

**Files:**
- Create: `tests/integration/test_product_smoke.py`
- Create: `examples/demo-corpus/project-overview.md`
- Create: `examples/demo-corpus/evaluation-notes.txt`

**Interfaces:**
- Consumes: complete fake-provider app.
- Produces: one repeatable upload-to-report integration test and bundled demo documents.

- [ ] **Step 1: Write the end-to-end failing test**

```python
# tests/integration/test_product_smoke.py
def test_fake_product_flow_uploads_runs_streams_and_reports(fake_app_client) -> None:
    upload = fake_app_client.post(
        "/api/documents",
        files=[("files", ("overview.md", b"The project requires traceable citations.", "text/markdown"))],
    )
    document_id = upload.json()["documents"][0]["id"]
    created = fake_app_client.post(
        "/api/research",
        json={"question": "What does the project require?", "document_ids": [document_id]},
    )
    run_id = created.json()["run_id"]
    fake_app_client.app.state.run_tasks.wait(run_id, timeout=5)
    result = fake_app_client.get(f"/api/research/{run_id}").json()
    assert result["run"]["status"] == "completed"
    assert result["report"]["citations"][0]["filename"] == "overview.md"
```

- [ ] **Step 2: Add a schema-aware fake-provider factory**

Create `build_demo_fake_provider()` in `app/providers/fake.py`. Its structured response callback must:

- return a one-query `ResearchPlan` for any question;
- extract valid `Evidence ID:` values from the Researcher prompt and use the first one in a `Finding`;
- return sufficient Critique when at least one valid ID exists;
- return a one-finding DraftReport using the same ID.

This makes the offline demonstration accept arbitrary uploaded text while remaining deterministic and grounded.

- [ ] **Step 3: Run full product tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/integration/test_product_smoke.py tests/api tests/static -q
```

Expected: PASS without `.env`, network, or API key.

- [ ] **Step 4: Commit the integrated product**

```powershell
git add app/providers/fake.py examples tests/integration/test_product_smoke.py
git commit -m "test: verify complete offline product flow"
```
