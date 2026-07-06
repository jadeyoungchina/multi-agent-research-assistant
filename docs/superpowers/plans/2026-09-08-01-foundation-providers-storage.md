# Foundation, Providers, and Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish typed configuration, domain contracts, deterministic test providers, real OpenAI-compatible providers, and SQLite persistence used by every later subsystem.

**Architecture:** Keep domain models free of framework and vendor dependencies. Provider implementations satisfy Protocol interfaces and accept injectable SDK clients for isolated tests. SQLite access is centralized behind small repositories using JSON and NumPy BLOB serialization.

**Tech Stack:** Python 3.11, Pydantic 2.8.2, pydantic-settings 2.4.0, OpenAI SDK 1.43.0, Tenacity 8.5.0, SQLite, NumPy 1.26.4, pytest 8.3.2.

**Spec:** `docs/superpowers/specs/2026-09-08-multi-agent-research-assistant-design.md`

## Global Constraints

- Default real chat provider is DashScope with model `qwen3.7-flash`.
- API keys come only from environment variables and never appear in logs or committed fixtures.
- Domain modules cannot import FastAPI, LangGraph, OpenAI SDK, or SQLite.
- Provider and storage tests must run without network access or an API key.
- All timestamps are timezone-aware UTC values in storage and are rendered in local time only at presentation boundaries.
- SQLite foreign keys and WAL mode are enabled on every connection.
- This phase must leave `pytest tests/config tests/domain tests/providers tests/storage -q` passing.

---

### Task 0: License and Upstream Provenance Baseline

**Files:**
- Create: `LICENSE`
- Create: `THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `docs/upstream-analysis.md`
- Create: `tests/release/test_attribution.py`

**Interfaces:**
- Produces: repository-wide non-commercial license boundary and a machine-checked provenance registry.
- Consumes: `D:\AIProjects\GenAI_Agents\LICENSE` at commit `4c95ae14cc2462c442b5c064cccd74430d02bc46`.

- [ ] **Step 1: Write the failing attribution test**

```python
# tests/release/test_attribution.py
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = "c9877e4d8788a0bb97502348dca8fbd78a6eaa73db0638221b3cb67422d30177"


def normalized_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return sha256(text.encode("utf-8")).hexdigest()


def test_upstream_license_is_vendored_verbatim() -> None:
    assert normalized_sha(ROOT / "LICENSE") == EXPECTED
    assert normalized_sha(ROOT / "THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt") == EXPECTED


def test_notice_pins_reused_notebooks() -> None:
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "4c95ae14cc2462c442b5c064cccd74430d02bc46" in notice
    for name in (
        "scientific_paper_agent_langgraph.ipynb",
        "document_intake_agent_langgraph.ipynb",
        "EU_Green_Compliance_FAQ_Bot.ipynb",
        "multi_agent_collaboration_system.ipynb",
        "trace_based_agent_evaluation.ipynb",
    ):
        assert name in notice
```

- [ ] **Step 2: Run the test and observe missing files**

Run: `.venv\Scripts\python.exe -m pytest tests/release/test_attribution.py -v`

Expected: FAIL because the license and notice files do not exist.

- [ ] **Step 3: Vendor the license verbatim and record provenance**

Use `apply_patch` to create both license files with the exact content of `D:\AIProjects\GenAI_Agents\LICENSE`. Create `THIRD_PARTY_NOTICES.md` with this table:

| Upstream file | Source cells | First available | Local targets | Treatment |
|---|---:|---|---|---|
| `scientific_paper_agent_langgraph.ipynb` | 13, 15, 19, 21 | 2024-11-17 | workflow, Planner, Critic | adapted |
| `document_intake_agent_langgraph.ipynb` | 13, 15, 19, 23 | 2026-07-03 | document routing | adapted |
| `EU_Green_Compliance_FAQ_Bot.ipynb` | 21, 25, 36 | 2024-11-17 | chunk/retrieval/query fusion | adapted |
| `multi_agent_collaboration_system.ipynb` | 6, 11–21 | 2024-09-09 | sequential baseline | adapted/concept-only by file |
| `trace_based_agent_evaluation.ipynb` | 5, 9, 13, 15, 19 | 2026-08-28 | trace evaluator | adapted |

State that the project is independent, non-commercial, not affiliated with or endorsed by Nir Diamant, and that all changes are documented per target file.

- [ ] **Step 4: Document the exact reuse audit**

`docs/upstream-analysis.md` must list for every source notebook:

- code or design retained;
- defects or unsafe behavior removed;
- local files receiving adapted logic;
- contributors named by the upstream history where known;
- whether the reuse is `adapted` or `concept-only`.

- [ ] **Step 5: Verify the attribution boundary**

Run: `.venv\Scripts\python.exe -m pytest tests/release/test_attribution.py -v`

Expected: PASS and both normalized license hashes equal `c9877e4d8788a0bb97502348dca8fbd78a6eaa73db0638221b3cb67422d30177`.

- [ ] **Step 6: Commit the license foundation**

```powershell
git add LICENSE THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt THIRD_PARTY_NOTICES.md docs/upstream-analysis.md tests/release/test_attribution.py
git commit -m "chore: establish licensed project foundation"
```

---

### Task 1: Project Configuration and Dependency Contract

**Files:**
- Modify: `requirements/project-requirements.txt`
- Modify: `.env.example`
- Modify: `.gitignore`
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `tests/config/test_settings.py`

**Interfaces:**
- Produces: `Settings`, `get_settings(env_file: Path | None = None) -> Settings`.
- Consumes: environment variables only.

- [ ] **Step 1: Write failing configuration tests**

```python
# tests/config/test_settings.py
from pathlib import Path

from app.config import Settings, get_settings


def test_settings_default_to_dashscope_qwen(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "app.db",
    )
    assert settings.chat_provider == "dashscope"
    assert settings.dashscope_chat_model == "qwen3.7-flash"
    assert settings.max_revision_iterations == 2
    assert settings.max_workflow_steps == 24
    assert settings.max_upload_file_bytes < settings.max_upload_total_bytes


def test_get_settings_reads_explicit_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHAT_PROVIDER=fake\nDASHSCOPE_CHAT_MODEL=qwen3.7-flash\n",
        encoding="utf-8",
    )
    assert get_settings(env_file).chat_provider == "fake"


def test_secret_values_are_hidden_from_repr() -> None:
    settings = Settings(_env_file=None, dashscope_api_key="secret-value")
    assert "secret-value" not in repr(settings)
```

- [ ] **Step 2: Run the focused tests and observe the missing module failure**

Run: `.venv\Scripts\python.exe -m pytest tests/config/test_settings.py -v`

Expected: FAIL because `app.config` does not exist.

- [ ] **Step 3: Add exact project dependencies and test configuration**

Append to `requirements/project-requirements.txt`:

```text
pydantic-settings==2.4.0
pypdf==4.3.1
python-multipart==0.0.9
```

Create `pyproject.toml`:

```toml
[project]
name = "multi-agent-research-assistant"
version = "1.0.0"
requires-python = ">=3.11,<3.13"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-ra"
markers = [
  "live: requires external API credentials",
]
```

- [ ] **Step 4: Implement typed settings**

```python
# app/config.py
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Multi-Agent Research Assistant"
    app_env: Literal["development", "test", "production"] = "development"
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    database_path: Path = Path("data/research_assistant.db")
    max_upload_file_bytes: int = 20 * 1024 * 1024
    max_upload_total_bytes: int = 50 * 1024 * 1024

    chat_provider: Literal["dashscope", "openai", "fake"] = "dashscope"
    embedding_provider: Literal["dashscope", "openai", "fake"] = "dashscope"
    dashscope_api_key: SecretStr | None = Field(default=None, repr=False)
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_chat_model: str = "qwen3.7-flash"
    dashscope_embedding_model: str = "text-embedding-v4"
    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    provider_timeout_seconds: float = 60.0
    provider_max_retries: int = 2
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_batch_size: int = 32
    retrieval_top_k: int = 6
    retrieval_candidate_k: int = 20
    retrieval_rrf_k: int = 60
    retrieval_min_similarity: float = 0.15
    max_query_expansions: int = 4
    max_revision_iterations: int = 2
    max_workflow_steps: int = 24
    worker_count: int = 2


def get_settings(env_file: Path | None = None) -> Settings:
    return Settings(_env_file=env_file if env_file is not None else ".env")
```

Update `.env.example` with the exact public names and blank secrets:

```text
CHAT_PROVIDER=dashscope
EMBEDDING_PROVIDER=dashscope
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL=qwen3.7-flash
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_PATH=data/research_assistant.db
UPLOAD_DIR=data/uploads
MAX_UPLOAD_FILE_BYTES=20971520
MAX_UPLOAD_TOTAL_BYTES=52428800
```

Add `data/`, `logs/`, `evaluation/results/*`, and `!.gitkeep` exceptions to `.gitignore`.

- [ ] **Step 5: Install project dependencies and run tests**

Run:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements/project-requirements.txt
.venv\Scripts\python.exe -m pytest tests/config/test_settings.py -v
```

Expected: all three tests PASS.

- [ ] **Step 6: Commit the configuration boundary**

```powershell
git add requirements/project-requirements.txt .env.example .gitignore pyproject.toml app/__init__.py app/config.py tests/config/test_settings.py
git commit -m "build: define project configuration and dependencies"
```

---

### Task 2: Domain Models and Validation Rules

**Files:**
- Create: `app/domain/__init__.py`
- Create: `app/domain/documents.py`
- Create: `app/domain/providers.py`
- Create: `app/domain/research.py`
- Create: `app/domain/runs.py`
- Create: `app/domain/errors.py`
- Create: `tests/domain/test_models.py`

**Interfaces:**
- Produces: all shared models documented in the roadmap plus `ResearchRun`, `RunEvent`, and stable domain exceptions.
- Consumes: Pydantic only.

- [ ] **Step 1: Write failing model invariant tests**

```python
# tests/domain/test_models.py
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.domain.documents import EvidenceChunk
from app.domain.research import Finding, ResearchPlan
from app.domain.runs import ResearchRun


def test_plan_requires_at_least_one_search_query() -> None:
    with pytest.raises(ValidationError):
        ResearchPlan(
            objective="compare evidence",
            subquestions=["what changed?"],
            search_queries=[],
            completion_criteria=["cite a source"],
        )


def test_finding_requires_supporting_evidence() -> None:
    with pytest.raises(ValidationError):
        Finding(claim="claim", supporting_evidence_ids=[], confidence="high")


def test_chunk_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        EvidenceChunk(
            id="c1", document_id="d1", filename="a.md", page_number=None,
            chunk_index=0, content="   ", content_sha256="abc",
        )


def test_run_is_timezone_aware() -> None:
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    assert run.created_at.tzinfo is UTC
    assert run.status == "queued"
```

- [ ] **Step 2: Run the model tests and observe import failures**

Run: `.venv\Scripts\python.exe -m pytest tests/domain/test_models.py -v`

Expected: FAIL because `app.domain` does not exist.

- [ ] **Step 3: Implement document and provider models**

```python
# app/domain/documents.py
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


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
    page_count: int = Field(ge=0)
    error_message: str | None = None
    created_at: datetime

    @classmethod
    def new(cls, filename: str, media_type: str, sha256: str) -> "DocumentRecord":
        return cls(
            id=uuid4().hex,
            filename=filename,
            media_type=media_type,
            sha256=sha256,
            storage_path="",
            status="processing",
            page_count=0,
            created_at=datetime.now(UTC),
        )


class EvidenceChunk(BaseModel):
    id: str
    document_id: str
    filename: str
    page_number: int | None
    chunk_index: int = Field(ge=0)
    content: str
    content_sha256: str
    embedding_model: str | None = None
    score: float | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk content must not be blank")
        return value
```

Implement `app/domain/providers.py` with the exact Protocol and metadata contract from the roadmap. Use `Field(default_factory=TokenUsage)` for `ProviderMetadata.usage`.

- [ ] **Step 4: Implement research, run, and error models**

```python
# app/domain/research.py
from typing import Literal
from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    objective: str
    subquestions: list[str] = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1)
    completion_criteria: list[str] = Field(min_length=1)


class Finding(BaseModel):
    claim: str
    supporting_evidence_ids: list[str] = Field(min_length=1)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class ResearchSynthesis(BaseModel):
    findings: list[Finding]
    unresolved_questions: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    sufficient: bool
    reason: str
    evidence_gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)


class ReportFinding(BaseModel):
    heading: str
    narrative: str
    evidence_ids: list[str] = Field(min_length=1)


class DraftReport(BaseModel):
    title: str
    summary: str
    findings: list[ReportFinding]
    limitations: list[str] = Field(default_factory=list)
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
```

```python
# app/domain/runs.py
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


RunStatus = Literal["queued", "running", "completed", "failed"]


class ResearchRun(BaseModel):
    id: str
    question: str
    document_ids: list[str]
    status: RunStatus
    current_stage: str
    last_completed_stage: str | None = None
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
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def new(cls, question: str, document_ids: list[str], provider: str, model: str) -> "ResearchRun":
        now = datetime.now(UTC)
        return cls(
            id=uuid4().hex, question=question, document_ids=document_ids,
            status="queued", current_stage="planner", iteration=0,
            provider=provider, model=model, created_at=now, updated_at=now,
        )


class RunEvent(BaseModel):
    sequence: int = Field(ge=1)
    run_id: str
    stage: str
    event_type: Literal["created", "started", "completed", "retry", "failed", "finished"]
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
```

Define `ConfigurationError`, `ProviderError`, `DocumentError`, `RetrievalError`, `WorkflowError`, and `CitationError` in `app/domain/errors.py`, each carrying a stable `code` attribute.

- [ ] **Step 5: Run focused and complete domain tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/domain/test_models.py -v
.venv\Scripts\python.exe -m pytest tests/config tests/domain -q
```

Expected: PASS.

- [ ] **Step 6: Commit the domain contract**

```powershell
git add app/domain tests/domain
git commit -m "feat: define research domain contracts"
```

---

### Task 3: Deterministic Fake Providers

**Files:**
- Create: `app/providers/__init__.py`
- Create: `app/providers/fake.py`
- Create: `tests/providers/test_fake_provider.py`

**Interfaces:**
- Consumes: `ChatMessage`, `ChatProvider`, `EmbeddingProvider`, `ProviderMetadata`.
- Produces: `FakeChatProvider`, `DeterministicEmbeddingProvider`.

- [ ] **Step 1: Write failing deterministic-provider tests**

```python
# tests/providers/test_fake_provider.py
from app.domain.providers import ChatMessage
from app.domain.research import ResearchPlan
from app.providers.fake import DeterministicEmbeddingProvider, FakeChatProvider


def test_fake_chat_returns_queued_structured_response() -> None:
    expected = ResearchPlan(
        objective="answer",
        subquestions=["q1"],
        search_queries=["query"],
        completion_criteria=["cite evidence"],
    )
    provider = FakeChatProvider(structured_responses=[expected])
    actual, metadata = provider.generate_structured(
        [ChatMessage(role="user", content="question")], ResearchPlan
    )
    assert actual == expected
    assert metadata.provider == "fake"


def test_fake_embedding_is_stable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=16)
    first, first_metadata = provider.embed_query("同一段文本")
    second, second_metadata = provider.embed_query("同一段文本")
    assert first == second
    assert round(sum(value * value for value in first), 7) == 1.0
    assert first_metadata.model == second_metadata.model == "fake-hash-16"
```

- [ ] **Step 2: Run tests and verify missing implementations**

Run: `.venv\Scripts\python.exe -m pytest tests/providers/test_fake_provider.py -v`

Expected: FAIL because `app.providers.fake` does not exist.

- [ ] **Step 3: Implement queue-backed chat responses**

```python
# app/providers/fake.py
from collections import deque
from hashlib import sha256
from math import sqrt
from time import perf_counter
from typing import Sequence, TypeVar

from pydantic import BaseModel

from app.domain.providers import ChatMessage, ProviderMetadata

T = TypeVar("T", bound=BaseModel)


class FakeChatProvider:
    def __init__(self, text_responses: Sequence[str] = (), structured_responses: Sequence[BaseModel] = ()) -> None:
        self._text = deque(text_responses)
        self._structured = deque(structured_responses)

    def generate(self, messages: Sequence[ChatMessage]) -> tuple[str, ProviderMetadata]:
        started = perf_counter()
        if not self._text:
            raise AssertionError("no fake text response queued")
        value = self._text.popleft()
        return value, ProviderMetadata(
            provider="fake", model="fake-chat", latency_ms=int((perf_counter() - started) * 1000)
        )

    def generate_structured(self, messages: Sequence[ChatMessage], schema: type[T]) -> tuple[T, ProviderMetadata]:
        started = perf_counter()
        if not self._structured:
            raise AssertionError(f"no fake response queued for {schema.__name__}")
        value = self._structured.popleft()
        if not isinstance(value, schema):
            raise AssertionError(f"expected {schema.__name__}, got {type(value).__name__}")
        return value, ProviderMetadata(
            provider="fake", model="fake-chat", latency_ms=int((perf_counter() - started) * 1000)
        )
```

- [ ] **Step 4: Implement normalized deterministic embeddings**

```python
class DeterministicEmbeddingProvider:
    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return f"fake-hash-{self._dimensions}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        encoded = text.casefold().encode("utf-8")
        for offset in range(0, len(encoded), 4):
            digest = sha256(encoded[offset:offset + 4]).digest()
            index = int.from_bytes(digest[:2], "big") % self._dimensions
            vector[index] += -1.0 if digest[2] & 1 else 1.0
        norm = sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> tuple[list[list[float]], ProviderMetadata]:
        started = perf_counter()
        vectors = [self._embed(text) for text in texts]
        return vectors, ProviderMetadata(
            provider="fake", model=self.model_name,
            latency_ms=int((perf_counter() - started) * 1000),
        )

    def embed_query(self, text: str) -> tuple[list[float], ProviderMetadata]:
        started = perf_counter()
        return self._embed(text), ProviderMetadata(
            provider="fake", model=self.model_name,
            latency_ms=int((perf_counter() - started) * 1000),
        )
```

- [ ] **Step 5: Run provider tests**

Run: `.venv\Scripts\python.exe -m pytest tests/providers/test_fake_provider.py -v`

Expected: PASS.

- [ ] **Step 6: Commit fake providers**

```powershell
git add app/providers tests/providers/test_fake_provider.py
git commit -m "feat: add deterministic test providers"
```

---

### Task 4: OpenAI-Compatible Chat and Embedding Providers

**Files:**
- Create: `app/providers/openai_compatible.py`
- Create: `app/providers/factory.py`
- Create: `tests/providers/test_openai_compatible.py`
- Create: `tests/providers/test_factory.py`

**Interfaces:**
- Consumes: `Settings`, provider Protocols, domain errors.
- Produces: `OpenAICompatibleChatProvider`, `OpenAICompatibleEmbeddingProvider`, `build_chat_provider(settings)`, `build_embedding_provider(settings)`.

- [ ] **Step 1: Write failing SDK-adapter tests with an injected client**

```python
# tests/providers/test_openai_compatible.py
from types import SimpleNamespace

from app.domain.providers import ChatMessage
from app.domain.research import Critique
from app.providers.openai_compatible import OpenAICompatibleChatProvider


class FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"sufficient":true,"reason":"ok","evidence_gaps":[],"follow_up_queries":[]}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="qwen3.7-flash",
        )


def test_structured_completion_is_validated() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    provider = OpenAICompatibleChatProvider(
        client=client, provider_name="dashscope", model="qwen3.7-flash", max_retries=0
    )
    result, metadata = provider.generate_structured(
        [ChatMessage(role="user", content="review")], Critique
    )
    assert result.sufficient is True
    assert metadata.usage.total_tokens == 15
```

Add a second test whose fake client raises twice and succeeds on the third call; assert `metadata.retries == 2`. Add a third test returning invalid JSON on every call and assert `ProviderError.code == "provider_invalid_response"` without including an API key in its text. Add embedding tests that assert ordered vectors, model/token/latency/retry metadata, exponential retry, and normalized `provider_embedding_failed` errors.

- [ ] **Step 2: Run focused tests and observe failure**

Run: `.venv\Scripts\python.exe -m pytest tests/providers/test_openai_compatible.py -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the chat adapter**

```python
# app/providers/openai_compatible.py
import json
from time import perf_counter, sleep
from typing import Sequence, TypeVar

from pydantic import BaseModel, ValidationError
from openai import APIError, APITimeoutError

from app.domain.errors import ProviderError
from app.domain.providers import ChatMessage, ProviderMetadata, TokenUsage

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleChatProvider:
    def __init__(self, client, provider_name: str, model: str, max_retries: int = 2) -> None:
        self.client = client
        self.provider_name = provider_name
        self.model = model
        self.max_retries = max_retries

    def _request(self, messages: Sequence[ChatMessage], json_mode: bool):
        kwargs = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**kwargs)

    def generate(self, messages: Sequence[ChatMessage]) -> tuple[str, ProviderMetadata]:
        return self._complete_with_retries(messages, None)

    def generate_structured(self, messages: Sequence[ChatMessage], schema: type[T]) -> tuple[T, ProviderMetadata]:
        content, metadata = self._complete_with_retries(messages, schema)
        return schema.model_validate_json(content), metadata

    def _complete_with_retries(self, messages, schema):
        started = perf_counter()
        last_error: Exception | None = None
        failure_code = "provider_request_failed"
        request_messages = list(messages)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request(request_messages, json_mode=schema is not None)
                content = response.choices[0].message.content or ""
                if schema is not None:
                    schema.model_validate_json(content)
                usage = getattr(response, "usage", None)
                return content, ProviderMetadata(
                    provider=self.provider_name,
                    model=getattr(response, "model", self.model),
                    latency_ms=int((perf_counter() - started) * 1000),
                    retries=attempt,
                    usage=TokenUsage(
                        prompt_tokens=getattr(usage, "prompt_tokens", 0),
                        completion_tokens=getattr(usage, "completion_tokens", 0),
                        total_tokens=getattr(usage, "total_tokens", 0),
                    ),
                )
            except ValidationError as exc:
                last_error = exc
                failure_code = "provider_invalid_response"
                if attempt < self.max_retries:
                    request_messages = [
                        *messages,
                        ChatMessage(
                            role="user",
                            content=f"Return valid JSON matching schema {schema.model_json_schema()}.",
                        ),
                    ]
                    sleep(0.25 * (2 ** attempt))
            except (APITimeoutError, TimeoutError) as exc:
                last_error = exc
                failure_code = "provider_timeout"
                if attempt < self.max_retries:
                    sleep(0.25 * (2 ** attempt))
            except APIError as exc:
                last_error = exc
                failure_code = "provider_request_failed"
                if attempt < self.max_retries:
                    sleep(0.25 * (2 ** attempt))
            except Exception as exc:
                last_error = exc
                failure_code = "provider_request_failed"
                if attempt < self.max_retries:
                    sleep(0.25 * (2 ** attempt))
        raise ProviderError(failure_code, f"provider request failed: {type(last_error).__name__}")
```

The final implementation must append a schema-specific repair instruction on retries after `ValidationError`; it must not include raw credentials or full provider responses in exceptions.

- [ ] **Step 4: Implement embeddings and provider factories**

```python
class OpenAICompatibleEmbeddingProvider:
    def __init__(self, client, provider_name: str, model: str, max_retries: int = 2) -> None:
        self.client = client
        self.provider_name = provider_name
        self._model = model
        self.max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> tuple[list[list[float]], ProviderMetadata]:
        started = perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(model=self._model, input=list(texts))
                usage = getattr(response, "usage", None)
                vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
                return vectors, ProviderMetadata(
                    provider=self.provider_name,
                    model=getattr(response, "model", self._model),
                    latency_ms=int((perf_counter() - started) * 1000),
                    retries=attempt,
                    usage=TokenUsage(prompt_tokens=getattr(usage, "prompt_tokens", 0), total_tokens=getattr(usage, "total_tokens", 0)),
                )
            except Exception as exc:
                if attempt == self.max_retries:
                    raise ProviderError("provider_embedding_failed", f"embedding request failed: {type(exc).__name__}") from exc
                sleep(0.25 * (2 ** attempt))

    def embed_query(self, text: str) -> tuple[list[float], ProviderMetadata]:
        vectors, metadata = self.embed_documents([text])
        return vectors[0], metadata
```

```python
# app/providers/factory.py
from openai import OpenAI

from app.config import Settings
from app.domain.errors import ConfigurationError
from app.providers.fake import DeterministicEmbeddingProvider, FakeChatProvider
from app.providers.openai_compatible import OpenAICompatibleChatProvider, OpenAICompatibleEmbeddingProvider


def _secret_value(secret, name: str) -> str:
    if secret is None or not secret.get_secret_value():
        raise ConfigurationError("missing_api_key", f"{name} is not configured")
    return secret.get_secret_value()


def build_chat_provider(settings: Settings):
    if settings.chat_provider == "fake":
        return FakeChatProvider()
    if settings.chat_provider == "dashscope":
        client = OpenAI(
            api_key=_secret_value(settings.dashscope_api_key, "DASHSCOPE_API_KEY"),
            base_url=settings.dashscope_base_url,
            timeout=settings.provider_timeout_seconds,
        )
        return OpenAICompatibleChatProvider(client, "dashscope", settings.dashscope_chat_model, settings.provider_max_retries)
    client = OpenAI(
        api_key=_secret_value(settings.openai_api_key, "OPENAI_API_KEY"),
        base_url=settings.openai_base_url,
        timeout=settings.provider_timeout_seconds,
    )
    return OpenAICompatibleChatProvider(client, "openai", settings.openai_chat_model, settings.provider_max_retries)
```

Implement `build_embedding_provider` with the same provider selection, model-specific settings, timeout, and `provider_max_retries`.

- [ ] **Step 5: Run provider and configuration regressions**

Run: `.venv\Scripts\python.exe -m pytest tests/config tests/providers -q`

Expected: PASS without network access.

- [ ] **Step 6: Commit provider adapters**

```powershell
git add app/providers tests/providers
git commit -m "feat: add OpenAI-compatible model providers"
```

---

### Task 5: SQLite Database and Schema

**Files:**
- Create: `app/storage/__init__.py`
- Create: `app/storage/database.py`
- Create: `app/storage/schema.sql`
- Create: `tests/storage/test_database.py`

**Interfaces:**
- Produces: `Database.connect()`, `Database.initialize()`, `Database.transaction()`.
- Consumes: `Settings.database_path`.

- [ ] **Step 1: Write failing database initialization tests**

```python
# tests/storage/test_database.py
from pathlib import Path

from app.storage.database import Database


def test_initialize_creates_expected_tables(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()
    with database.connect() as connection:
        names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"documents", "chunks", "research_runs", "run_events", "reports"} <= names


def test_connection_enables_foreign_keys_and_wal(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.initialize()
    with database.connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
```

- [ ] **Step 2: Run tests and observe failure**

Run: `.venv\Scripts\python.exe -m pytest tests/storage/test_database.py -v`

Expected: FAIL because storage modules do not exist.

- [ ] **Step 3: Create the exact schema**

```sql
-- app/storage/schema.sql
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    storage_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('processing', 'ready', 'failed')),
    page_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    embedding_model TEXT,
    embedding BLOB,
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    document_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    last_completed_stage TEXT,
    iteration INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    model_call_count INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    evidence_sufficient INTEGER CHECK(evidence_sufficient IN (0, 1)),
    state_json TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    stage TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, sequence)
);

CREATE TABLE IF NOT EXISTS reports (
    run_id TEXT PRIMARY KEY REFERENCES research_runs(id) ON DELETE CASCADE,
    report_json TEXT NOT NULL,
    markdown TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON run_events(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_runs_status ON research_runs(status);
```

- [ ] **Step 4: Implement the connection wrapper**

```python
# app/storage/database.py
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.executescript(schema)

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
```

- [ ] **Step 5: Run database tests**

Run: `.venv\Scripts\python.exe -m pytest tests/storage/test_database.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the database schema**

```powershell
git add app/storage/database.py app/storage/schema.sql tests/storage/test_database.py
git commit -m "feat: add SQLite application schema"
```

---

### Task 6: Document and Run Repositories

**Files:**
- Create: `app/storage/serialization.py`
- Create: `app/storage/documents.py`
- Create: `app/storage/runs.py`
- Create: `tests/storage/test_document_repository.py`
- Create: `tests/storage/test_run_repository.py`

**Interfaces:**
- Produces: `DocumentRepository`, `RunRepository`, `vector_to_blob()`, `blob_to_vector()`.
- Consumes: domain models and `Database`.

- [ ] **Step 1: Write failing repository round-trip tests**

```python
# tests/storage/test_document_repository.py
import numpy as np

from app.domain.documents import EvidenceChunk
from app.storage.documents import DocumentRepository


def test_chunk_embedding_round_trip(database, document_record) -> None:
    repository = DocumentRepository(database)
    repository.add_document(document_record)
    chunk = EvidenceChunk(
        id="chunk-1", document_id=document_record.id, filename=document_record.filename,
        page_number=1, chunk_index=0, content="evidence", content_sha256="hash",
        embedding_model="fake-hash-3",
    )
    repository.replace_chunks(document_record.id, [(chunk, [0.1, 0.2, 0.3])])
    stored = repository.list_chunks([document_record.id])
    assert stored[0][0] == chunk
    assert np.allclose(stored[0][1], [0.1, 0.2, 0.3])
```

```python
# tests/storage/test_run_repository.py
from app.domain.runs import ResearchRun
from app.storage.runs import RunRepository


def test_run_state_and_events_round_trip(database) -> None:
    repository = RunRepository(database)
    run = ResearchRun.new("question", ["d1"], provider="fake", model="fake")
    state = {"run_id": run.id, "next_stage": "planner", "provider_metrics": []}
    repository.create(run, state)
    repository.start_stage(run.id, state, "planner")
    completed = {**state, "next_stage": "retriever", "last_completed_stage": "planner"}
    repository.complete_stage(
        run.id, completed, "retriever", "planner", 0, [], {"attempt": 1}
    )
    assert repository.get_state(run.id)["next_stage"] == "retriever"
    assert repository.list_events(run.id, after_sequence=0)[0].sequence == 1
```

Add rollback-injection tests proving `complete_ingestion()`, `start_stage()`,
`complete_stage()`, and `fail_stage()` never expose a partial multi-row
mutation. The final-stage case asserts one report, `status="completed"`, the
expected aggregate token/call/retry values, `evidence_sufficient`, and exactly
one terminal `finished` event. Add a two-thread repository test that appends
events for separate runs and verifies both transactions commit without sharing
a connection; event sequence allocation (`MAX(sequence) + 1`) occurs inside
`BEGIN IMMEDIATE`.

- [ ] **Step 2: Run tests and observe repository failures**

Run: `.venv\Scripts\python.exe -m pytest tests/storage/test_document_repository.py tests/storage/test_run_repository.py -v`

Expected: FAIL because repositories do not exist.

- [ ] **Step 3: Implement stable vector serialization**

```python
# app/storage/serialization.py
import numpy as np


def vector_to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes(order="C")


def blob_to_vector(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).astype(float).tolist()
```

- [ ] **Step 4: Implement repositories with explicit methods**

`DocumentRepository` must implement:

```python
def add_document(self, document: DocumentRecord) -> None: ...
def get_document(self, document_id: str) -> DocumentRecord | None: ...
def get_document_by_sha256(self, sha256: str) -> DocumentRecord | None: ...
def list_documents(self) -> list[DocumentRecord]: ...
def update_document(self, document: DocumentRecord) -> None: ...
def update_status(
    self,
    document_id: str,
    status: str,
    page_count: int,
    error_message: str | None = None,
) -> None: ...
def replace_chunks(self, document_id: str, chunks: list[tuple[EvidenceChunk, list[float]]]) -> None: ...
def complete_ingestion(
    self,
    document: DocumentRecord,
    chunks: list[tuple[EvidenceChunk, list[float]]],
) -> None: ...
def fail_ingestion(self, document_id: str, error_message: str) -> None: ...
def list_chunks(self, document_ids: list[str]) -> list[tuple[EvidenceChunk, list[float]]]: ...
def delete_document(self, document_id: str) -> None: ...
```

`RunRepository` must implement:

```python
def create(self, run: ResearchRun, state: dict) -> None: ...
def get(self, run_id: str) -> ResearchRun | None: ...
def get_state(self, run_id: str) -> dict: ...
def start_stage(self, run_id: str, state: dict, stage: str) -> RunEvent: ...
def get_report(self, run_id: str) -> ResearchReport | None: ...
def append_event(self, run_id: str, stage: str, event_type: str, payload: dict) -> RunEvent: ...
def list_events(self, run_id: str, after_sequence: int) -> list[RunEvent]: ...
def list_incomplete(self) -> list[ResearchRun]: ...
def complete_stage(
    self,
    run_id: str,
    state: dict,
    current_stage: str,
    last_completed_stage: str,
    iteration: int,
    provider_metrics: list[dict],
    event_payload: dict,
    report: ResearchReport | None = None,
) -> RunEvent: ...
def fail_stage(self, run_id: str, state: dict, stage: str, code: str, message: str) -> RunEvent: ...
```

`list_incomplete()` returns only `queued` and `running` runs, ordered by
`created_at`; failed and completed runs are never auto-rescheduled.

Use `model_dump(mode="json")`, `model_validate_json`, `json.dumps(..., ensure_ascii=False)`, ISO-8601 timestamps, parameterized SQL, and one transaction per multi-row mutation. `create()` inserts the run, initial state, and `created` event atomically. `complete_ingestion()` atomically replaces all chunks and marks the document ready; `fail_ingestion()` deletes partial chunks and marks the document failed. `start_stage()` atomically saves the unchanged start snapshot, marks the run running, and appends the started event. `complete_stage()` atomically saves the merged checkpoint, enforces `current_stage == state["next_stage"]`, records `last_completed_stage`, sets `model_call_count = len(provider_metrics)`, and recomputes token/retry totals by summing the cumulative metadata before appending the completed event. For the citation-validator stage it also upserts the report, marks the run completed, sets `completed_at` and `evidence_sufficient`, and appends a terminal `finished` event in the same transaction. `fail_stage()` atomically preserves the last start snapshot, marks failure, and appends the failed event. Every method opens its own `Database.transaction()` connection so calls are safe from the API thread pool.

- [ ] **Step 5: Run all storage tests**

Run: `.venv\Scripts\python.exe -m pytest tests/storage -q`

Expected: PASS.

- [ ] **Step 6: Run the phase gate and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/config tests/domain tests/providers tests/storage -q`

Expected: PASS.

```powershell
git add app/storage tests/storage
git commit -m "feat: persist documents and research runs"
```
