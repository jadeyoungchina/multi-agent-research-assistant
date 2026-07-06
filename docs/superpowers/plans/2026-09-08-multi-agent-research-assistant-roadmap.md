# Multi-Agent Research Assistant Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild and publish a complete non-commercial multi-agent research assistant with traceable local-document RAG, DashScope Qwen support, a FastAPI web interface, reproducible evaluation, and transparent reconstructed Git history.

**Architecture:** Build one modular Python application around a LangGraph workflow. FastAPI serves both JSON/SSE APIs and a dependency-free browser UI; SQLite stores documents, embeddings, run snapshots, events, and reports. Upstream notebook implementations are extracted and hardened where suitable, while missing production boundaries are implemented behind explicit interfaces.

**Tech Stack:** Python 3.11, LangGraph 0.2.18, LangChain 0.2.x, OpenAI Python SDK 1.43.0, FastAPI 0.112.2, Pydantic 2.8.2, SQLite, NumPy 1.26.4, PyPDF, vanilla HTML/CSS/JavaScript, pytest 8.3.2.

**Spec:** `docs/superpowers/specs/2026-09-08-multi-agent-research-assistant-design.md`

## Global Constraints

- The project is for personal learning and non-commercial portfolio display.
- Pin upstream attribution to `NirDiamant/GenAI_Agents@4c95ae14cc2462c442b5c064cccd74430d02bc46`.
- Prefer adapting suitable upstream Agent, workflow, and utility implementations; implement only missing or unsafe parts.
- Record the source notebook, cell numbers, changes, and license in every adapted source file and in `THIRD_PARTY_NOTICES.md`.
- Keep the upstream custom non-commercial license and do not describe the repository as MIT, Apache, or commercially usable.
- Default real chat provider is DashScope OpenAI-compatible API with model `qwen3.7-flash`.
- Chat and embedding models are independently configurable; no API key is hard-coded, logged, returned, or committed.
- v1.0 accepts PDF, Markdown, and TXT only; it does not perform web search.
- v1.0 is a local single-user application without authentication or multi-tenancy.
- Use SQLite for documents, chunks, embeddings, run state, events, and reports.
- Use deterministic Fake providers for default tests and offline demonstrations.
- Every generated report citation must resolve to evidence retrieved during that run.
- Critic revision is limited to two iterations and the graph has a total-step limit.
- Uploaded files, SQLite databases, generated indices, `.env`, logs, and evaluation results are not committed.
- Implementation follows TDD: failing test, observed failure, minimal implementation, passing focused test, broader regression test, commit.
- The final reconstructed Git history is explicitly disclosed as reconstructed after accidental deletion.
- Final commit timestamps use `Asia/Shanghai` and occur after 20:00 on the approved July-to-September milestone dates.

---

## Plan Set

These documents group work by subsystem. Execute individual tasks in the
dependency-safe milestone order below; do not run each whole document in
numeric order. Every milestone leaves the repository in a runnable and
independently testable state.

1. [`2026-09-08-01-foundation-providers-storage.md`](2026-09-08-01-foundation-providers-storage.md)
   - Project configuration and dependencies.
   - Domain contracts.
   - Fake, DashScope, and OpenAI-compatible providers.
   - SQLite schema and repositories.

2. [`2026-09-08-02-document-retrieval.md`](2026-09-08-02-document-retrieval.md)
   - Safe local document loading.
   - Deterministic chunking.
   - Embedding persistence and local vector search.
   - Query fusion and document ingestion service.

3. [`2026-09-08-03-agent-workflow.md`](2026-09-08-03-agent-workflow.md)
   - Five Agent nodes.
   - LangGraph state and bounded routing.
   - Citation validation.
   - Stage snapshots, restart recovery, and research service.

4. [`2026-09-08-04-api-web-interface.md`](2026-09-08-04-api-web-interface.md)
   - FastAPI application and error model.
   - Document and research APIs.
   - Background execution and SSE.
   - Native browser interface.

5. [`2026-09-08-05-evaluation-documentation-release.md`](2026-09-08-05-evaluation-documentation-release.md)
   - Trace evaluator and four benchmark configurations.
   - Thirty-case dataset and reports.
   - Documentation, attribution, CI, security checks, and Qwen smoke test.
   - Safe Git history reconstruction, GitHub push, and `v1.0.0` tag.

## Shared Interface Contract

The following names are stable across all five plans. Later plans must consume them without renaming.

```python
# app/domain/providers.py
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ProviderMetadata(BaseModel):
    provider: str
    model: str
    latency_ms: int
    retries: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)

T = TypeVar("T", bound=BaseModel)

class ChatProvider(Protocol):
    def generate(self, messages: Sequence[ChatMessage]) -> tuple[str, ProviderMetadata]: ...
    def generate_structured(
        self, messages: Sequence[ChatMessage], schema: type[T]
    ) -> tuple[T, ProviderMetadata]: ...

class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...
    def embed_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], ProviderMetadata]: ...
    def embed_query(self, text: str) -> tuple[list[float], ProviderMetadata]: ...
```

```python
# app/domain/documents.py
class LoadedPage(BaseModel):
    page_number: int | None
    text: str

class DocumentRecord(BaseModel):
    id: str
    filename: str
    media_type: str
    sha256: str
    storage_path: str
    status: Literal["processing", "ready", "failed"]
    page_count: int
    error_message: str | None = None
    created_at: datetime

class EvidenceChunk(BaseModel):
    id: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    content: str
    content_sha256: str
    embedding_model: str | None = None
    score: float | None = None
```

```python
# app/domain/research.py
class ResearchPlan(BaseModel):
    objective: str
    subquestions: list[str]
    search_queries: list[str]
    completion_criteria: list[str]

class Finding(BaseModel):
    claim: str
    supporting_evidence_ids: list[str]
    conflicting_evidence_ids: list[str] = []
    confidence: Literal["low", "medium", "high"]

class ResearchSynthesis(BaseModel):
    findings: list[Finding]
    unresolved_questions: list[str] = []

class Critique(BaseModel):
    sufficient: bool
    reason: str
    evidence_gaps: list[str] = []
    follow_up_queries: list[str] = []

class ReportFinding(BaseModel):
    heading: str
    narrative: str
    evidence_ids: list[str]

class DraftReport(BaseModel):
    title: str
    summary: str
    findings: list[ReportFinding]
    limitations: list[str] = []
    markdown: str

class Citation(BaseModel):
    evidence_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    excerpt: str

class ResearchReport(BaseModel):
    title: str
    summary: str
    findings: list[ReportFinding]
    limitations: list[str]
    citations: list[Citation]
    markdown: str
    evidence_sufficient: bool

# app/domain/runs.py
class ResearchRun(BaseModel):
    id: str
    question: str
    document_ids: list[str]
    status: Literal["queued", "running", "completed", "failed"]
    current_stage: str
    last_completed_stage: str | None
    iteration: int
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_call_count: int = 0
    retry_count: int = 0
    evidence_sufficient: bool | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
```

```python
# app/retrieval/contracts.py
class RetrievalBatch(BaseModel):
    evidence: list[EvidenceChunk]
    provider_metrics: list[ProviderMetadata] = Field(default_factory=list)
```

```python
# app/workflow/state.py
class WorkflowState(TypedDict, total=False):
    run_id: str
    question: str
    document_ids: list[str]
    plan: dict[str, Any]
    evidence: list[dict[str, Any]]
    synthesis: dict[str, Any]
    critique: dict[str, Any]
    draft: dict[str, Any]
    report: dict[str, Any]
    iteration: int
    next_stage: str
    last_completed_stage: str | None
    provider_metrics: list[dict[str, Any]]
    errors: list[dict[str, str]]
```

```python
# app/services/research.py
class ResearchService:
    def create_run(self, question: str, document_ids: list[str]) -> ResearchRun: ...
    def execute(self, run_id: str) -> ResearchReport: ...
    def get_run(self, run_id: str) -> ResearchRun: ...
    def get_report(self, run_id: str) -> ResearchReport | None: ...
    def list_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]: ...
    def list_incomplete_ids(self) -> list[str]: ...
```

`WorkflowState.next_stage` is the resumable graph cursor. The repository must
atomically keep `ResearchRun.current_stage == state["next_stage"]` after every
completed stage and record the completed node in `last_completed_stage`.
Embedding metadata collected during retrieval is appended to
`WorkflowState.provider_metrics`; `ResearchRun` token/call/retry totals are
recomputed from that cumulative list in the same checkpoint transaction.

## Verification Gates

- After plan 1: `pytest tests/config tests/domain tests/providers tests/storage -q` passes.
- After plan 2: `pytest tests/retrieval tests/services/test_document_service.py -q` passes and a sample document can be ingested without an API key.
- After plan 3: `pytest tests/agents tests/workflow tests/services/test_research_service.py -q` passes and the complete Fake Provider workflow produces a citation-valid report.
- After plan 4: `pytest tests/api -q` passes and the browser smoke path can upload, run, receive events, and render a report.
- After plan 5: `pytest -q` and the offline benchmark pass; optional live Qwen smoke test passes when `DASHSCOPE_API_KEY` is present.
- Before push: `git diff --check`, secret scan, ignored-data check, clean worktree, Git history audit, and remote lease verification all pass.

## Milestone Execution Order

The detailed plans are grouped by subsystem for readability, but implementation follows this dependency-safe milestone order so the reconstructed commit graph is chronological:

1. Existing environment, approved design, implementation plans, then plan 1 Task 0.
2. Plan 1 Tasks 1, 2, 3, 5, and 6 for configuration, domain, fake providers, and persistence.
3. Plan 3 Tasks 1–5 using a fake retrieval implementation behind the retrieval Protocol.
4. Plan 2 Tasks 1–3 for document loading, chunking, and embedding persistence.
5. Plan 2 Tasks 4–5 for vector search and fused retrieval.
6. Plan 3 Task 6 for checkpointing and recovery against the completed retrieval layer.
7. Plan 1 Task 4 and plan 4 Tasks 1–4 for real Providers, FastAPI, research jobs, and SSE.
8. Plan 5 Tasks 1–4 for dataset, metrics, variants, reports, and gates.
9. Plan 4 Tasks 5–6 for the browser dashboard and full offline smoke path.
10. Plan 5 Tasks 5–7 for documentation, live Qwen verification, history reconstruction, and publication.

## Approved Reconstructed Milestones

| Date and time (`+08:00`) | Deliverable |
|---|---|
| 2026-07-06 20:18 | Environment, license, and upstream analysis |
| 2026-07-13 21:07 | Project skeleton, configuration, and domain models |
| 2026-07-22 22:16 | Baseline Agents and LangGraph workflow |
| 2026-07-30 20:43 | Document ingestion and chunking |
| 2026-08-08 21:26 | Vector retrieval and citation tracking |
| 2026-08-17 22:09 | Critic revision loop and recovery |
| 2026-08-25 20:51 | Qwen/OpenAI Provider and API |
| 2026-09-01 21:34 | Evaluation framework and benchmark |
| 2026-09-05 22:12 | Web interface and full integration |
| 2026-09-08 21:40 | Tests, documentation, and v1.0.0 |
