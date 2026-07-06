# Evaluation, Documentation, and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thirty-case, four-variant benchmark; generate reproducible reports; complete documentation and CI; verify Qwen; and publish the project to GitHub as `v1.0.0`.

**Architecture:** Evaluation adapters normalize each workflow into one trace schema. Deterministic metrics score retrieval, citations, answer coverage, latency, token usage, and failures. Release checks verify attribution, repository hygiene, tests, and remote state before publication.

**Tech Stack:** Python 3.11, Pydantic 2.8.2, pytest 8.3.2, JSONL, CSV, Markdown, GitHub Actions, Git.

**Spec:** `docs/superpowers/specs/2026-09-08-multi-agent-research-assistant-design.md`

## Global Constraints

- Adapt trace structures, nearest-rank percentile, failure isolation, and suite aggregation from `trace_based_agent_evaluation.ipynb` cells 5, 9, 13, 15, and 19.
- Never pass gold answers or expected evidence into an evaluated workflow.
- The synthetic benchmark contains exactly 30 cases and is clearly labeled as synthetic.
- Compare `baseline_llm`, `llm_rag`, `single_agent_rag`, and `multi_agent_rag` on identical cases.
- Record provider, model, prompt version, corpus hash, Git commit, latency, tokens, calls, Critic loops, and errors without recording secrets.
- Default tests and CI use fake providers only.
- Fake multi-Agent citation precision must be 1.0.
- Quantitative README claims must point to generated results or be labeled illustrative.

---

### Task 1: Benchmark Models, Corpus, and Thirty Cases

**Files:**
- Create: `app/evaluation/__init__.py`
- Create: `app/evaluation/models.py`
- Create: `app/evaluation/dataset.py`
- Create: `benchmarks/cases.jsonl`
- Create: `benchmarks/quality-gates.json`
- Create: `benchmarks/corpus/solar-storage.md`
- Create: `benchmarks/corpus/urban-water.md`
- Create: `benchmarks/corpus/public-health.md`
- Create: `benchmarks/corpus/education-technology.md`
- Create: `benchmarks/corpus/software-reliability.md`
- Create: `benchmarks/corpus/research-governance.md`
- Create: `tests/evaluation/test_dataset.py`

**Interfaces:**
- Produces: `WorkflowVariant`, `EvidenceExpectation`, `BenchmarkCase`, `load_benchmark_cases()`, `validate_benchmark_corpus()`.

- [ ] **Step 1: Write failing dataset tests**

```python
# tests/evaluation/test_dataset.py
from pathlib import Path

from app.evaluation.dataset import load_benchmark_cases, validate_benchmark_corpus


def test_default_dataset_has_thirty_sequential_unique_cases() -> None:
    cases = load_benchmark_cases(Path("benchmarks/cases.jsonl"))
    assert len(cases) == 30
    assert [case.id for case in cases] == [f"BENCH-{number:03d}" for number in range(1, 31)]


def test_expected_phrases_exist_in_declared_sources() -> None:
    cases = load_benchmark_cases(Path("benchmarks/cases.jsonl"))
    validate_benchmark_corpus(cases, Path("benchmarks/corpus"))
```

Add tests rejecting duplicate/non-sequential IDs, `../` paths, missing sources, blank key points, and more than two Critic loops.

- [ ] **Step 2: Run tests and observe failure**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_dataset.py -v`

- [ ] **Step 3: Implement strict models and loading**

```python
# app/evaluation/models.py
from enum import StrEnum
from pydantic import BaseModel, Field


class WorkflowVariant(StrEnum):
    BASELINE_LLM = "baseline_llm"
    LLM_RAG = "llm_rag"
    SINGLE_AGENT_RAG = "single_agent_rag"
    MULTI_AGENT_RAG = "multi_agent_rag"


class EvidenceExpectation(BaseModel):
    source_file: str
    contains: str = Field(min_length=1)


class BenchmarkCase(BaseModel):
    id: str = Field(pattern=r"^BENCH-\d{3}$")
    question: str = Field(min_length=3)
    source_files: list[str] = Field(min_length=1)
    expected_evidence: list[EvidenceExpectation] = Field(min_length=1)
    answer_key_points: list[str] = Field(min_length=1)
    max_critic_loops: int = Field(default=2, ge=0, le=2)
    latency_budget_ms: int | None = Field(default=None, gt=0)
```

`load_benchmark_cases()` parses one JSON object per nonblank line and verifies exact sequential IDs. `validate_benchmark_corpus()` resolves each source under `corpus_dir`, rejects escapes, and casefold-checks every expected phrase.

- [ ] **Step 4: Create the exact synthetic corpus**

Every file starts with `> Synthetic benchmark data for software evaluation; not a real-world research claim.` and includes these facts verbatim:

| File | Facts |
|---|---|
| `solar-storage.md` | `12 MW solar array`; `24 MWh battery`; `peak grid purchases fell by 18%`; `curtailment fell from 9% to 3%`; `simple payback period is 7.5 years` |
| `urban-water.md` | `15,000 households`; `leakage fell from 22% to 14%`; `1.8 million cubic metres per year`; `alerts within 10 minutes`; `pilot cost was CNY 6.4 million` |
| `public-health.md` | `24 townships`; `screening rose from 61% to 84%`; `waiting time fell from 46 to 29 minutes`; `follow-up completion rose from 72% to 88%`; `12% of follow-up records were self-reported` |
| `education-technology.md` | `18 schools`; `3,200 students`; `completion rose from 68% to 81%`; `12 hours of teacher training`; `network interruptions affected 7% of sessions` |
| `software-reliability.md` | `weekly to daily deployments`; `change failure rate fell from 14% to 8%`; `recovery time fell from 95 to 41 minutes`; `10% canary traffic for 30 minutes`; `99.9% monthly availability permits about 43.8 minutes of downtime` |
| `research-governance.md` | `raw model outputs are retained for 90 days`; `API keys are removed before logs are written`; `two reviewers are required before publication`; `dataset versions are pinned with SHA-256`; `every external claim must cite at least one valid Evidence ID` |

- [ ] **Step 5: Create all thirty cases**

Create four single-source questions per corpus file as `BENCH-001` through `BENCH-024`, asking for facts 1–4 respectively. Each case names the source, exact expected phrase, normalized answer point, `max_critic_loops=2`, and `latency_budget_ms=5000`.

Create these cross-source cases with `latency_budget_ms=8000`:

| ID | Question | Expected phrases |
|---|---|---|
| BENCH-025 | What percentage indicators improved in the solar and urban-water pilots? | `peak grid purchases fell by 18%`; `leakage fell from 22% to 14%` |
| BENCH-026 | By how many percentage points did public-health screening and education completion improve? | `screening rose from 61% to 84%`; `completion rose from 68% to 81%` |
| BENCH-027 | What time controls were used in urban-water monitoring and software deployment? | `alerts within 10 minutes`; `10% canary traffic for 30 minutes` |
| BENCH-028 | What quantified limitations were reported for public health and education technology? | `12% of follow-up records were self-reported`; `network interruptions affected 7% of sessions` |
| BENCH-029 | Which two governance measures support later audit and reproduction? | `raw model outputs are retained for 90 days`; `dataset versions are pinned with SHA-256` |
| BENCH-030 | When reporting the software availability target, what downtime and citation rule must accompany it? | `43.8 minutes of downtime`; `every external claim must cite at least one valid Evidence ID` |

- [ ] **Step 6: Add the quality gate and verify**

```json
{
  "multi_agent_rag": {
    "minimum_success_rate": 1.0,
    "minimum_retrieval_recall_at_5": 0.9,
    "minimum_citation_precision": 1.0,
    "minimum_evidence_coverage": 0.9,
    "minimum_answer_key_point_coverage": 0.8,
    "maximum_failure_rate": 0.0
  }
}
```

Only `multi_agent_rag` is a release gate in v1.0. The other three variants are
comparison baselines: their full metrics and failures must still appear in
JSON, CSV, and Markdown, but they do not block publication.

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_dataset.py -q`

Expected: PASS.

- [ ] **Step 7: Commit dataset and corpus**

```powershell
git add app/evaluation/models.py app/evaluation/dataset.py benchmarks tests/evaluation/test_dataset.py
git commit -m "feat: add reproducible research benchmark"
```

---

### Task 2: Trace Scoring and Aggregate Metrics

**Files:**
- Create: `app/evaluation/trace.py`
- Create: `app/evaluation/metrics.py`
- Create: `tests/evaluation/test_trace.py`
- Create: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Produces: `EvidenceSnapshot`, `EvaluationTrace`, `CaseScore`, `score_case()`, `percentile()`, `aggregate_scores()`.

- [ ] **Step 1: Write failing trace and metric tests**

```python
# tests/evaluation/test_trace.py
def test_failed_trace_scores_zero_without_aborting(case) -> None:
    trace = EvaluationTrace.failed(case.id, WorkflowVariant.MULTI_AGENT_RAG, "provider_timeout")
    score = score_case(case, trace)
    assert score.success is False
    assert score.citation_precision == 0.0


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([10, 20, 30, 40], 0.95) == 40
```

```python
# tests/evaluation/test_metrics.py
def test_grounding_metrics_have_exact_values(case, trace) -> None:
    score = score_case(case, trace)
    assert score.retrieval_recall_at_5 == 0.5
    assert score.citation_precision == 0.5
    assert score.evidence_coverage == 0.5
```

- [ ] **Step 2: Run tests and observe failure**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_trace.py tests/evaluation/test_metrics.py -v`

- [ ] **Step 3: Adapt the upstream evaluator without target leakage**

```python
# app/evaluation/trace.py
# Adapted from trace_based_agent_evaluation.ipynb cells 5, 9, 13, 15, 19.
# Changes: gold expectations stay evaluator-only; latency is measured externally;
# retrieval, citation, answer coverage, token, loop, and error metrics are added.

class EvidenceSnapshot(BaseModel):
    id: str
    source_file: str
    page_number: int | None
    chunk_index: int
    text: str


class EvaluationTrace(BaseModel):
    case_id: str
    variant: WorkflowVariant
    status: Literal["success", "failed"]
    answer: str
    retrieved_evidence: list[EvidenceSnapshot] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    critic_loops: int = Field(ge=0)
    provider: str
    model: str
    error_code: str | None = None
```

The workflow receives only its question and source document IDs. The evaluator owns `perf_counter()` timing and overwrites self-reported latency.

- [ ] **Step 4: Implement deterministic metrics**

- Tokenize each CJK character and Latin/number spans using `[a-z0-9]+`.
- For each answer key point, take the maximum token F1 against answer sentences; average those maxima.
- Count key points with F1 `>=0.60` for answer coverage.
- Compute retrieval recall from expected phrases found in the first five correct-source evidence texts.
- Compute citation precision from valid unique IDs divided by all unique cited IDs.
- Compute evidence coverage from expected phrases covered by valid cited evidence.
- Use nearest-rank `ceil(p*n)-1` for p50 and p95.

- [ ] **Step 5: Run tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_trace.py tests/evaluation/test_metrics.py -q`

Expected: PASS.

```powershell
git add app/evaluation/trace.py app/evaluation/metrics.py tests/evaluation/test_trace.py tests/evaluation/test_metrics.py
git commit -m "feat: add trace-based evaluation metrics"
```

---

### Task 3: Four Evaluation Workflow Variants

**Files:**
- Create: `app/evaluation/workflows.py`
- Create: `tests/evaluation/test_workflows.py`

**Interfaces:**
- Consumes: providers, retrieval service, `ResearchService`.
- Produces: `prepare_benchmark_documents()` and four implementations of `run(case_id, question, source_files) -> EvaluationTrace`.

- [ ] **Step 1: Write failing adapter tests**

```python
# tests/evaluation/test_workflows.py
def test_all_variants_return_normalized_traces(workflow_factory, case) -> None:
    traces = [workflow_factory(variant).run(case.id, case.question, case.source_files) for variant in WorkflowVariant]
    assert [trace.variant for trace in traces] == list(WorkflowVariant)
    assert all(trace.case_id == case.id for trace in traces)
    assert traces[0].retrieved_evidence == []
    assert traces[3].critic_loops <= 2


def test_corpus_files_are_ingested_once_and_mapped_to_document_ids(corpus_factory) -> None:
    mapping = corpus_factory.prepare()
    assert mapping["solar-storage.md"].startswith("doc")
    assert corpus_factory.document_service.ingest_count("solar-storage.md") == 1
```

Add tests proving gold expectations are never passed to adapters, unknown citation IDs fail a trace, and one adapter failure does not change others.

- [ ] **Step 2: Implement exact adapter behavior**

Before constructing variants, `prepare_benchmark_documents(cases, corpus_dir, document_service)` reads every unique declared source under the confined corpus root, ingests it once, and returns `dict[source_file, document_id]`. Every RAG variant resolves `case.source_files` through this same mapping; source filenames are never passed to `ResearchService` as document IDs.

1. `BaselineLlmWorkflow`: one chat generation, no retrieval or citations.
2. `LlmRagWorkflow`: retrieve with the original question once, use `RetrievalBatch.evidence` for grounding, and include its embedding metadata plus one chat generation in the trace.
3. `SingleAgentRagWorkflow`: structured query expansion, fused retrieval, then one structured grounded answer; normalize both embedding and chat metadata into the trace.
4. `MultiAgentRagWorkflow`: create and execute a `ResearchService` run, then normalize stored state/report and copy `ResearchRun` token, call, retry, iteration, and evidence-sufficiency fields into the trace.

```python
class QueryPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)


class GroundedAnswer(BaseModel):
    answer_markdown: str
    cited_evidence_ids: list[str]
```

Grounded answer IDs must be a subset of retrieved IDs or the trace fails with `invalid_citation`.

- [ ] **Step 3: Run tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/evaluation/test_workflows.py -q`

Expected: PASS.

```powershell
git add app/evaluation/workflows.py tests/evaluation/test_workflows.py
git commit -m "feat: compare four research workflows"
```

---

### Task 4: Runner, Reports, CLI, and Quality Gate

**Files:**
- Create: `app/evaluation/runner.py`
- Create: `app/evaluation/reporters.py`
- Create: `app/evaluation/quality_gate.py`
- Create: `app/evaluation/cli.py`
- Create: `tests/evaluation/test_runner.py`
- Create: `tests/evaluation/test_reporters.py`
- Create: `tests/evaluation/test_cli.py`

**Interfaces:**
- Produces: `evaluate_benchmark()`, `write_reports()`, `enforce_quality_gate()`, and `python -m app.evaluation.cli`.

- [ ] **Step 1: Write failing runner tests**

```python
# tests/evaluation/test_runner.py
def test_runner_produces_case_variant_matrix(cases, workflows) -> None:
    report = evaluate_benchmark(cases, workflows)
    assert len(report.traces) == len(cases) * 4


def test_one_failure_does_not_abort_remaining_runs(cases, workflows) -> None:
    workflows[WorkflowVariant.LLM_RAG].fail_case_id = cases[0].id
    report = evaluate_benchmark(cases, workflows)
    assert len(report.traces) == len(cases) * 4
    assert sum(trace.status == "failed" for trace in report.traces) == 1
```

- [ ] **Step 2: Implement deterministic execution and aggregation**

Iterate cases by ID and variants in enum order. Record UTC run time, `git rev-parse HEAD`, corpus SHA-256, safe provider/model configuration, prompt version, traces, scores, p50/p95, success/failure rate, and mean token/call/loop counts.

- [ ] **Step 3: Write reporter tests**

```python
def test_reporters_write_all_formats(tmp_path, report) -> None:
    paths = write_reports(report, tmp_path)
    assert {path.name for path in paths} == {"results.json", "results.csv", "report.md"}
    assert "single run does not establish a general conclusion" in (tmp_path / "report.md").read_text("utf-8")
```

- [ ] **Step 4: Implement formats, gate, and CLI**

- JSON stores the typed report.
- CSV stores one case/variant per row.
- Markdown contains four-variant comparison, failures, configuration, hashes, and reproducibility warning.
- Gate reports all failed thresholds in one error.
- CLI exit codes are `0` success, `2` invalid dataset, `3` failed gate, `4` execution error.

Run command:

```powershell
.venv\Scripts\python.exe -m app.evaluation.cli --dataset benchmarks/cases.jsonl --corpus benchmarks/corpus --provider fake --variants all --output-dir artifacts/evaluation/fake --enforce-gate
```

- [ ] **Step 5: Verify and commit**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/evaluation -q
.venv\Scripts\python.exe -m app.evaluation.cli --dataset benchmarks/cases.jsonl --corpus benchmarks/corpus --provider fake --variants all --output-dir artifacts/evaluation/fake --enforce-gate
```

Expected: 120 traces, three output files, and gate PASS.

```powershell
git add app/evaluation tests/evaluation benchmarks
git commit -m "feat: evaluate four research workflows"
```

---

### Task 5: Documentation, CI, and Repository Hygiene

**Files:**
- Create: `README.md`
- Create: `README.zh-CN.md`
- Create: `CHANGELOG.md`
- Create: `docs/architecture.md`
- Create: `docs/experiments.md`
- Create: `docs/technical-report.md`
- Create: `docs/release-checklist.md`
- Create: `.github/workflows/ci.yml`
- Create: `scripts/verify_repository.py`
- Create: `tests/release/test_documentation.py`
- Create: `tests/release/test_repository_hygiene.py`
- Modify: `.env.example`
- Modify: `.gitignore`
- Modify: `docs/superpowers/specs/2026-09-08-multi-agent-research-assistant-design.md`

**Interfaces:**
- Produces: complete user/developer documentation, CI, and `python scripts/verify_repository.py`.

- [ ] **Step 1: Write failing documentation tests**

```python
# tests/release/test_documentation.py
from pathlib import Path


def test_readmes_document_license_and_usage() -> None:
    english = Path("README.md").read_text(encoding="utf-8")
    chinese = Path("README.zh-CN.md").read_text(encoding="utf-8")
    for text in (english, chinese):
        lowered = text.casefold()
        assert "non-commercial" in lowered or "非商业" in text
        assert "qwen3.7-flash" in text


def test_required_documents_exist() -> None:
    for path in (
        "CHANGELOG.md", "docs/architecture.md", "docs/experiments.md",
        "docs/technical-report.md",
        "docs/release-checklist.md", "THIRD_PARTY_NOTICES.md",
    ):
        assert Path(path).is_file(), path
```

- [ ] **Step 2: Write the project documentation**

Both READMEs include features, architecture, offline fake quick start, DashScope setup, upload/research/SSE API examples, test commands, benchmark commands, local-data privacy, upstream attribution, independent-project disclaimer, and non-commercial restrictions.

`CHANGELOG.md` starts with `1.0.0 - 2026-09-08` and lists the delivered changes. `docs/experiments.md` does not claim real-model quality until a saved real run exists.

- [ ] **Step 3: Write repository-hygiene tests**

```python
# tests/release/test_repository_hygiene.py
def test_runtime_data_is_not_tracked() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    forbidden = (".env", "data/", "artifacts/", ".db", ".sqlite", ".sqlite3")
    assert not [path for path in tracked if path and any(token in path for token in forbidden)]


def test_example_environment_contains_no_secret() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith(("DASHSCOPE_API_KEY=", "OPENAI_API_KEY=")):
            assert line.endswith("=")
```

- [ ] **Step 4: Implement `scripts/verify_repository.py`**

The script must:

1. obtain tracked files with `git ls-files -z`;
2. reject `.env`, `data/`, `artifacts/`, `*.db`, `*.sqlite*`, uploaded documents, and generated evaluation output;
3. scan text files for non-placeholder OpenAI and DashScope key patterns;
4. permit only blank key assignments in `.env.example`;
5. verify `LICENSE` and `THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt` normalize to SHA-256 `c9877e4d8788a0bb97502348dca8fbd78a6eaa73db0638221b3cb67422d30177`;
6. validate all 30 benchmark cases and evidence phrases;
7. verify each `adapted` local file in `THIRD_PARTY_NOTICES.md` exists and contains the pinned upstream commit.

- [ ] **Step 5: Add CI**

```yaml
name: ci

on: [push, pull_request]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install -r requirements/upstream-requirements.txt
      - run: python -m pip install -r requirements/project-requirements.txt
      - run: python -m pytest -m "not live" -q
      - run: python scripts/verify_repository.py
      - run: python -m app.evaluation.cli --provider fake --variants all --output-dir artifacts/evaluation/fake --enforce-gate
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: fake-benchmark
          path: artifacts/evaluation/fake
```

- [ ] **Step 6: Run documentation and hygiene checks**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/release -q
.venv\Scripts\python.exe scripts/verify_repository.py
git diff --check
```

Expected: PASS and no tracked runtime data or secrets.

- [ ] **Step 7: Commit final documentation**

```powershell
git add README.md README.zh-CN.md CHANGELOG.md docs .github scripts/verify_repository.py tests/release .env.example .gitignore
git commit -m "docs: publish v1.0.0 documentation"
```

---

### Task 6: Optional Live Qwen Verification

**Files:**
- Create: `tests/live/test_dashscope_smoke.py`

**Interfaces:**
- Consumes: local untracked `.env` with `DASHSCOPE_API_KEY`.
- Produces: opt-in proof that `qwen3.7-flash` is reachable through the real Provider.

- [ ] **Step 1: Write the opt-in test**

```python
# tests/live/test_dashscope_smoke.py
import os
import pytest

from app.config import get_settings
from app.domain.providers import ChatMessage
from app.providers.factory import build_chat_provider


@pytest.mark.live
def test_dashscope_qwen_smoke() -> None:
    if os.getenv("RUN_LIVE_PROVIDER_TESTS") != "1":
        pytest.skip("live provider tests are opt-in")
    settings = get_settings()
    if settings.dashscope_api_key is None:
        pytest.skip("DASHSCOPE_API_KEY is not configured")
    provider = build_chat_provider(settings)
    text, metadata = provider.generate([ChatMessage(role="user", content="Reply with exactly: pong")])
    assert text.strip().casefold() == "pong"
    assert metadata.model == "qwen3.7-flash"
```

- [ ] **Step 2: Verify default skip behavior**

Run: `.venv\Scripts\python.exe -m pytest tests/live/test_dashscope_smoke.py -v`

Expected: SKIPPED unless the explicit live flag is set.

- [ ] **Step 3: Run the real smoke test without sharing the key**

The user places the key in the untracked `.env`; never request that it be pasted into chat.

Run:

```powershell
$env:RUN_LIVE_PROVIDER_TESTS='1'
.venv\Scripts\python.exe -m pytest tests/live/test_dashscope_smoke.py -m live -o addopts="" -v
Remove-Item Env:RUN_LIVE_PROVIDER_TESTS
```

Expected: PASS with a `pong` response and no key in output.

- [ ] **Step 4: Commit the opt-in smoke test**

```powershell
git add tests/live/test_dashscope_smoke.py
git commit -m "test: verify optional Qwen connectivity"
```

---

### Task 7: Full Verification and GitHub Publication

- [ ] Run the complete offline test suite.
- [ ] Run `python scripts/verify_repository.py`.
- [ ] Run the fake benchmark for all four variants and inspect its quality gate.
- [ ] Review the staged diff, attribution, and release checklist.
- [ ] Verify a clean worktree and review the current remote state.
- [ ] Push the verified main branch and intended release tag.
- [ ] Check remote CI and report the repository URL, commit ID, tag, and validation results.
