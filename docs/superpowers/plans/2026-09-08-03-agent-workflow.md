# Multi-Agent Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the five specialized Agent nodes, bounded Critic revision loop, deterministic citation validation, LangGraph orchestration, stage snapshots, and restartable research service.

**Architecture:** Each Agent is a small class that consumes typed state and returns a state update; it never reads environment variables, FastAPI, or SQLite directly. A graph factory wires the nodes and routes Critic feedback. A checkpoint wrapper persists the merged state and an event after every completed stage so a run can restart from `next_stage`.

**Tech Stack:** Python 3.11, LangGraph 0.2.18, Pydantic 2.8.2, SQLite repositories from plan 1, retrieval services from plan 2, pytest 8.3.2.

**Spec:** `docs/superpowers/specs/2026-09-08-multi-agent-research-assistant-design.md`

## Global Constraints

- Adapt the state, conditional routing, and judge loop from `scientific_paper_agent_langgraph.ipynb` cells 13, 15, 19, and 21.
- Adapt the role separation and final synthesis responsibility from `multi_agent_collaboration_system.ipynb` cells 6 and 11–21.
- Agent output is always validated by Pydantic before entering state.
- Every `Finding` and `ReportFinding` references only Evidence IDs present in the current run.
- Critic may request at most two additional retrieval rounds.
- Invoke the graph with `recursion_limit=24`; twelve node executions are required by the longest valid two-revision path in LangGraph 0.2.18.
- Reaching the loop limit does not mark evidence sufficient; the final report must retain an insufficiency limitation.
- Writer does not invent citation metadata and cannot cite evidence absent from state.
- Each successfully completed stage writes one checkpoint and one completed event before the next stage executes.
- A failed stage preserves the previous complete checkpoint and marks the run failed with a safe error.
- All tests use fake providers and an isolated temporary SQLite database.

---

### Task 1: Workflow State and Prompt Boundaries

**Files:**
- Create: `app/workflow/__init__.py`
- Create: `app/workflow/state.py`
- Create: `app/retrieval/__init__.py`
- Create: `app/retrieval/contracts.py`
- Create: `app/agents/__init__.py`
- Create: `app/agents/prompts.py`
- Create: `app/agents/common.py`
- Create: `tests/workflow/test_state.py`
- Create: `tests/agents/test_prompts.py`

**Interfaces:**
- Consumes: domain models from plan 1.
- Produces: `WorkflowState`, `initial_state()`, `EvidenceRetrieverProtocol`, `format_evidence()`, and prompt builders.

- [ ] **Step 1: Write failing state serialization tests**

```python
# tests/workflow/test_state.py
from app.workflow.state import initial_state, merge_state


def test_initial_state_starts_at_planner() -> None:
    state = initial_state("run1", "What changed?", ["doc1"])
    assert state["iteration"] == 0
    assert state["next_stage"] == "planner"
    assert state["evidence"] == []
    assert state["provider_metrics"] == []


def test_state_json_round_trip_preserves_writer_draft() -> None:
    state = initial_state("run1", "question", ["doc1"])
    updated = merge_state(state, {"next_stage": "citation_validator", "draft": {"title": "Draft"}})
    assert json.loads(json.dumps(updated, ensure_ascii=False))["draft"]["title"] == "Draft"
```

- [ ] **Step 2: Write failing evidence-format tests**

```python
# tests/agents/test_prompts.py
from app.agents.common import format_evidence


def test_format_evidence_includes_stable_ids_and_sources(chunk) -> None:
    rendered = format_evidence([chunk])
    assert f"Evidence ID: {chunk.id}" in rendered
    assert "Source: paper.pdf, page 2, chunk 4" in rendered
    assert chunk.content in rendered
```

- [ ] **Step 3: Run focused tests and observe failure**

Run: `.venv\Scripts\python.exe -m pytest tests/workflow/test_state.py tests/agents/test_prompts.py -v`

Expected: FAIL because workflow and Agent modules do not exist.

- [ ] **Step 4: Implement JSON-safe state helpers**

```python
# app/workflow/state.py
from copy import deepcopy
from typing import Any, TypedDict


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


def initial_state(run_id: str, question: str, document_ids: list[str]) -> WorkflowState:
    return WorkflowState(
        run_id=run_id,
        question=question,
        document_ids=list(document_ids),
        evidence=[],
        iteration=0,
        next_stage="planner",
        last_completed_stage=None,
        provider_metrics=[],
        errors=[],
    )


def merge_state(state: WorkflowState, updates: dict[str, Any]) -> WorkflowState:
    merged = deepcopy(dict(state))
    merged.update(deepcopy(updates))
    return WorkflowState(**merged)
```

Define the retrieval boundary before its concrete implementation exists:

```python
# app/retrieval/contracts.py
from typing import Protocol

from pydantic import BaseModel, Field

from app.domain.documents import EvidenceChunk
from app.domain.providers import ProviderMetadata


class RetrievalBatch(BaseModel):
    evidence: list[EvidenceChunk]
    provider_metrics: list[ProviderMetadata] = Field(default_factory=list)


class EvidenceRetrieverProtocol(Protocol):
    def retrieve(
        self,
        question: str,
        expansions: list[str],
        document_ids: list[str],
        top_k: int,
    ) -> RetrievalBatch: ...
```

- [ ] **Step 5: Implement prompt helpers with fixed grounding rules**

`app/agents/prompts.py` must expose:

```python
def planner_messages(question: str) -> list[ChatMessage]: ...
def researcher_messages(question: str, plan: ResearchPlan, evidence: list[EvidenceChunk]) -> list[ChatMessage]: ...
def critic_messages(question: str, plan: ResearchPlan, synthesis: ResearchSynthesis, evidence: list[EvidenceChunk]) -> list[ChatMessage]: ...
def writer_messages(question: str, synthesis: ResearchSynthesis, critique: Critique, evidence: list[EvidenceChunk]) -> list[ChatMessage]: ...
```

Every research, critic, and writer system prompt includes these exact rules:

```text
Use only the supplied evidence.
Refer to sources exclusively by their Evidence ID.
If evidence is absent or conflicting, state the limitation explicitly.
Never create an Evidence ID or source detail.
```

The Writer prompt additionally requires Markdown and the exact citation token
`[[cite:{Evidence ID}]]`; it forbids filenames, page numbers, and invented
reference labels because the deterministic validator supplies them.

- [ ] **Step 6: Run tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/workflow/test_state.py tests/agents/test_prompts.py -q`

Expected: PASS.

```powershell
git add app/workflow app/agents tests/workflow/test_state.py tests/agents/test_prompts.py
git commit -m "feat: define workflow state and grounded prompts"
```

---

### Task 2: Planner, Retriever, and Researcher Agents

**Files:**
- Create: `app/agents/planner.py`
- Create: `app/agents/retriever.py`
- Create: `app/agents/researcher.py`
- Create: `tests/agents/test_planner.py`
- Create: `tests/agents/test_retriever.py`
- Create: `tests/agents/test_researcher.py`

**Interfaces:**
- Consumes: `ChatProvider`, `EvidenceRetriever`, prompt helpers, typed state.
- Produces: callable Agent nodes returning `dict[str, Any]` updates.

- [ ] **Step 1: Write failing Planner tests**

```python
# tests/agents/test_planner.py
def test_planner_stores_validated_plan_and_metrics() -> None:
    plan = ResearchPlan(
        objective="compare",
        subquestions=["what agrees?", "what conflicts?"],
        search_queries=["agreement", "conflict"],
        completion_criteria=["cite every finding"],
    )
    provider = FakeChatProvider(structured_responses=[plan])
    update = PlannerAgent(provider)(initial_state("run", "question", ["doc"]))
    assert update["plan"] == plan.model_dump(mode="json")
    assert update["next_stage"] == "retriever"
    assert update["provider_metrics"][-1]["provider"] == "fake"
```

- [ ] **Step 2: Implement Planner from upstream structured decision pattern**

```python
# app/agents/planner.py
# Adapted from scientific_paper_agent_langgraph.ipynb cells 13 and 19.
# Changes: provider injection, ResearchPlan schema, no global model, no direct answer shortcut.

class PlannerAgent:
    def __init__(self, provider) -> None:
        self.provider = provider

    def __call__(self, state: WorkflowState) -> dict:
        plan, metadata = self.provider.generate_structured(
            planner_messages(state["question"]), ResearchPlan
        )
        return {
            "plan": plan.model_dump(mode="json"),
            "next_stage": "retriever",
            "provider_metrics": [*state.get("provider_metrics", []), metadata.model_dump(mode="json")],
        }
```

- [ ] **Step 3: Write and implement Retriever tests**

```python
# tests/agents/test_retriever.py
def test_retriever_uses_plan_and_critic_queries_once(fake_retriever) -> None:
    state = make_state(
        plan=ResearchPlan(...).model_dump(mode="json"),
        critique=Critique(
            sufficient=False, reason="gap", evidence_gaps=["missing"], follow_up_queries=["follow up"]
        ).model_dump(mode="json"),
    )
    update = RetrieverAgent(fake_retriever, top_k=6)(state)
    assert fake_retriever.calls[0].expansions == ["agreement", "conflict", "follow up"]
    assert update["next_stage"] == "researcher"
    assert update["provider_metrics"][-1]["model"] == "fake-hash-16"
```

`RetrieverAgent` reconstructs `ResearchPlan` and optional `Critique`, passes the question and combined query expansions to plan 2's `EvidenceRetriever`, merges `RetrievalBatch.evidence` with old evidence by ID, keeps the highest score, sorts by descending score then ID, and appends `RetrievalBatch.provider_metrics` to the state's cumulative provider metrics.

- [ ] **Step 4: Write failing Researcher grounding tests**

```python
# tests/agents/test_researcher.py
def test_researcher_rejects_unknown_evidence_id() -> None:
    synthesis = ResearchSynthesis(
        findings=[Finding(claim="claim", supporting_evidence_ids=["unknown"], confidence="high")]
    )
    provider = FakeChatProvider(structured_responses=[synthesis])
    with pytest.raises(CitationError, match="unknown_evidence_id"):
        ResearcherAgent(provider)(state_with_evidence("ev_valid"))
```

Add a positive test with supporting and conflicting Evidence IDs.

- [ ] **Step 5: Implement Researcher and deterministic grounding check**

```python
class ResearcherAgent:
    def __init__(self, provider) -> None:
        self.provider = provider

    def __call__(self, state: WorkflowState) -> dict:
        evidence = [EvidenceChunk.model_validate(item) for item in state["evidence"]]
        synthesis, metadata = self.provider.generate_structured(
            researcher_messages(
                state["question"], ResearchPlan.model_validate(state["plan"]), evidence
            ),
            ResearchSynthesis,
        )
        allowed = {item.id for item in evidence}
        referenced = {
            evidence_id
            for finding in synthesis.findings
            for evidence_id in [*finding.supporting_evidence_ids, *finding.conflicting_evidence_ids]
        }
        unknown = referenced - allowed
        if unknown:
            raise CitationError("unknown_evidence_id", f"researcher referenced {len(unknown)} unknown evidence IDs")
        return {
            "synthesis": synthesis.model_dump(mode="json"),
            "next_stage": "critic",
            "provider_metrics": [*state.get("provider_metrics", []), metadata.model_dump(mode="json")],
        }
```

- [ ] **Step 6: Run Agent tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/agents/test_planner.py tests/agents/test_retriever.py tests/agents/test_researcher.py -q`

Expected: PASS.

```powershell
git add app/agents/planner.py app/agents/retriever.py app/agents/researcher.py tests/agents
git commit -m "feat: add planning retrieval and research agents"
```

---

### Task 3: Critic and Bounded Revision Routing

**Files:**
- Create: `app/agents/critic.py`
- Create: `app/workflow/routing.py`
- Create: `tests/agents/test_critic.py`
- Create: `tests/workflow/test_routing.py`

**Interfaces:**
- Consumes: plan, evidence, synthesis, configured maximum iterations.
- Produces: `CriticAgent`, `route_after_critic()`.

- [ ] **Step 1: Write failing deterministic Critic tests**

```python
# tests/agents/test_critic.py
def test_critic_forces_revision_when_a_finding_has_no_valid_support() -> None:
    llm_value = Critique(sufficient=True, reason="looks good")
    provider = FakeChatProvider(structured_responses=[llm_value])
    state = state_with_invalid_finding()
    update = CriticAgent(provider)(state)
    critique = Critique.model_validate(update["critique"])
    assert critique.sufficient is False
    assert critique.evidence_gaps == ["one or more findings lack valid supporting evidence"]
```

Also test zero findings, plan criteria not covered, valid evidence preserved, and provider metadata appended.

- [ ] **Step 2: Implement Critic with deterministic checks and LLM judgment**

The deterministic audit calculates:

```python
valid_ids = {item.id for item in evidence}
supported = all(f.supporting_evidence_ids and set(f.supporting_evidence_ids) <= valid_ids for f in synthesis.findings)
has_findings = bool(synthesis.findings)
```

`CriticAgent` calls `ChatProvider.generate_structured(..., Critique)`, then applies
the deterministic audit. If either check fails, it returns an insufficient
`Critique` even when the LLM verdict was sufficient. It calls
`route_after_critic()`, returns the chosen `next_stage` and iteration, and
appends the provider metadata exactly once.

- [ ] **Step 3: Write routing boundary tests**

```python
# tests/workflow/test_routing.py
@pytest.mark.parametrize(
    ("iteration", "sufficient", "expected_stage", "expected_iteration"),
    [
        (0, False, "retriever", 1),
        (1, False, "retriever", 2),
        (2, False, "writer", 2),
        (0, True, "writer", 0),
    ],
)
def test_route_after_critic(iteration, sufficient, expected_stage, expected_iteration):
    assert route_after_critic(iteration, sufficient, max_iterations=2) == (
        expected_stage, expected_iteration
    )
```

- [ ] **Step 4: Implement routing without forced approval**

```python
# app/workflow/routing.py
def route_after_critic(iteration: int, sufficient: bool, max_iterations: int) -> tuple[str, int]:
    if sufficient:
        return "writer", iteration
    if iteration < max_iterations:
        return "retriever", iteration + 1
    return "writer", iteration
```

The implementation must preserve `Critique.sufficient=False` when the limit is reached.

- [ ] **Step 5: Run tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/agents/test_critic.py tests/workflow/test_routing.py -q`

Expected: PASS.

```powershell
git add app/agents/critic.py app/workflow/routing.py tests/agents/test_critic.py tests/workflow/test_routing.py
git commit -m "feat: add bounded evidence critique"
```

---

### Task 4: Writer and Deterministic Citation Validator

**Files:**
- Create: `app/agents/writer.py`
- Create: `app/workflow/citations.py`
- Create: `tests/agents/test_writer.py`
- Create: `tests/workflow/test_citations.py`

**Interfaces:**
- Consumes: synthesis, critique, evidence.
- Produces: `WriterAgent`, `validate_and_render_report()`, `CitationValidatorNode`.

- [ ] **Step 1: Write failing Writer tests**

```python
# tests/agents/test_writer.py
def test_writer_preserves_only_finding_evidence_ids() -> None:
    draft = DraftReport(
        title="Report",
        summary="Summary",
        findings=[ReportFinding(heading="Finding", narrative="Supported result", evidence_ids=["ev_good"])],
        markdown="# Report\n\n## Finding\nSupported result [[cite:ev_good]]",
    )
    provider = FakeChatProvider(structured_responses=[draft])
    update = WriterAgent(provider)(state_with_evidence("ev_good"))
    assert update["draft"]["findings"][0]["evidence_ids"] == ["ev_good"]
    assert update["next_stage"] == "citation_validator"
```

Add tests that unknown IDs raise `CitationError`, every `ReportFinding.evidence_ids` entry appears as a citation token in `draft.markdown`, and an insufficient Critique adds a limitation even when the provider omits it.

- [ ] **Step 2: Implement Writer as structured Markdown drafting**

Writer calls `ChatProvider.generate_structured(..., DraftReport)`, stores the validated draft under `state["draft"]`, and appends its `ProviderMetadata`. The draft includes Markdown written by the model. The only allowed inline citation syntax is `[[cite:{evidence_id}]]`. Writer validates that every token and every `ReportFinding.evidence_ids` value names evidence in state, but it does not add filenames, pages, reference numbers, anchors, or excerpts. Those are validator responsibilities.

- [ ] **Step 3: Write citation-validator tests**

```python
# tests/workflow/test_citations.py
def test_validator_expands_valid_ids_and_renders_markdown() -> None:
    report = validate_and_render_report(
        "run1", draft_with("ev_good"), [chunk(id="ev_good")], critique(sufficient=True)
    )
    assert report.citations[0].evidence_id == "ev_good"
    assert "[1](#citation-ev_good)" in report.markdown
    assert '<a id="citation-ev_good"></a>' in report.markdown
    assert report.evidence_sufficient is True


def test_validator_rejects_unknown_id_before_creating_report() -> None:
    with pytest.raises(CitationError, match="unknown_evidence_id"):
        validate_and_render_report(
            "run1", draft_with("ev_unknown"), [chunk(id="ev_good")], critique(sufficient=True)
        )
```

Also test malformed tokens, missing tokens for declared finding evidence, first-appearance ordering, deduplication, 320-character excerpts, page-less Markdown sources, citation metadata, and `evidence_sufficient=False` with an explicit insufficiency limitation after the revision limit.

- [ ] **Step 4: Implement deterministic rendering**

`validate_and_render_report(run_id, draft, evidence, critique)` parses complete tokens with `r"\[\[cite:([A-Za-z0-9_-]+)\]\]"` and rejects any leftover `[[cite:` fragment, unknown token, or `ReportFinding.evidence_ids` value omitted from Markdown. It orders citations by first token appearance, replaces tokens in `draft.markdown` with numbered local links, and appends the citation section. Markdown format:

```markdown
{draft.markdown with each [[cite:{evidence_id}]] replaced by [1](#citation-{evidence_id})}

## 局限
- {limitation}

## 引用
<a id="citation-{evidence_id}"></a>**[1] {filename}, page {page}, chunk {chunk_index}**
> {excerpt}
```

For Markdown/TXT sources omit `page`. Escape user-controlled filenames and excerpts before placing them in Markdown/HTML anchors.

`CitationValidatorNode.__call__(state)` reconstructs `DraftReport`, `Critique`, and
`EvidenceChunk` values, calls `validate_and_render_report()`, and returns
`{"report": report.model_dump(mode="json"), "next_stage": "completed"}`. It is
the only node allowed to create the final `ResearchReport`.

- [ ] **Step 5: Run tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/agents/test_writer.py tests/workflow/test_citations.py -q`

Expected: PASS.

```powershell
git add app/agents/writer.py app/workflow/citations.py tests/agents/test_writer.py tests/workflow/test_citations.py
git commit -m "feat: add citation-safe report writing"
```

---

### Task 5: LangGraph Assembly and Complete Fake Workflow

**Files:**
- Create: `app/workflow/graph.py`
- Create: `tests/workflow/test_graph.py`

**Interfaces:**
- Consumes: five Agent callables, validator callable, routing helper.
- Produces: `WorkflowDependencies`, `build_research_graph()`.

- [ ] **Step 1: Write failing graph-path tests**

```python
# tests/workflow/test_graph.py
def test_graph_runs_happy_path_once(dependencies) -> None:
    graph = build_research_graph(dependencies)
    result = graph.invoke(initial_state("run", "question", ["doc"]), {"recursion_limit": 24})
    assert dependencies.visited == [
        "planner", "retriever", "researcher", "critic", "writer", "citation_validator"
    ]
    assert result["next_stage"] == "completed"


def test_graph_retrieves_three_times_when_critic_exhausts_two_revisions(dependencies) -> None:
    dependencies.critic_results = [False, False, False]
    result = build_research_graph(dependencies).invoke(
        initial_state("run", "question", ["doc"]), {"recursion_limit": 24}
    )
    assert dependencies.visited.count("retriever") == 3
    assert Critique.model_validate(result["critique"]).sufficient is False
```

Add tests for one revision, direct sufficiency, and recursion-limit failure.

- [ ] **Step 2: Implement graph topology**

```python
# app/workflow/graph.py
# Adapted from scientific_paper_agent_langgraph.ipynb cells 15, 19, and 21.
# Changes: five explicit roles, local evidence retrieval, bounded revision,
# resume dispatch, deterministic citation validation, dependency injection.

from dataclasses import dataclass
from typing import Callable
from langgraph.graph import END, StateGraph

Node = Callable[[WorkflowState], dict]
NodeWrapper = Callable[[str, Node], Node]


@dataclass(frozen=True)
class WorkflowDependencies:
    planner: Node
    retriever: Node
    researcher: Node
    critic: Node
    writer: Node
    citation_validator: Node
    max_iterations: int = 2
    node_wrapper: NodeWrapper | None = None


def build_research_graph(deps: WorkflowDependencies):
    graph = StateGraph(WorkflowState)
    wrap = deps.node_wrapper or (lambda _stage, handler: handler)
    graph.add_node("planner", wrap("planner", deps.planner))
    graph.add_node("retriever", wrap("retriever", deps.retriever))
    graph.add_node("researcher", wrap("researcher", deps.researcher))
    graph.add_node("critic", wrap("critic", deps.critic))
    graph.add_node("writer", wrap("writer", deps.writer))
    graph.add_node("citation_validator", wrap("citation_validator", deps.citation_validator))
    graph.set_conditional_entry_point(lambda state: state["next_stage"], {
        "planner": "planner", "retriever": "retriever", "researcher": "researcher",
        "critic": "critic", "writer": "writer", "citation_validator": "citation_validator",
        "completed": END,
    })
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges("critic", _critic_route, {"retriever": "retriever", "writer": "writer"})
    graph.add_edge("writer", "citation_validator")
    graph.add_edge("citation_validator", END)
    return graph.compile()
```

Do not add an empty dispatch node: LangGraph 0.2.18 rejects a node that writes no state keys. The real `_critic_route` reads the Critique and delegates to `route_after_critic`; the Critic node update must set the incremented iteration and chosen `next_stage` before routing. Add a resume-path test whose initial `next_stage="researcher"` proves Planner and Retriever are skipped.

- [ ] **Step 3: Run graph and Agent regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/agents tests/workflow -q`

Expected: PASS.

- [ ] **Step 4: Commit the compiled workflow**

```powershell
git add app/workflow/graph.py tests/workflow/test_graph.py
git commit -m "feat: compile bounded multi-agent workflow"
```

---

### Task 6: Stage Checkpoints, Failure Handling, and Research Service

**Files:**
- Create: `app/workflow/checkpoints.py`
- Modify: `app/workflow/graph.py`
- Create: `app/services/research.py`
- Create: `tests/workflow/test_checkpoints.py`
- Create: `tests/services/test_research_service.py`
- Create: `tests/integration/test_workflow_recovery.py`

**Interfaces:**
- Consumes: graph, `RunRepository`, `DocumentRepository`, `ResearchRun`, `RunEvent`.
- Produces: `checkpointed_node()`, `ResearchService` matching the roadmap contract.

- [ ] **Step 1: Write failing checkpoint atomicity tests**

```python
# tests/workflow/test_checkpoints.py
def test_completed_stage_saves_merged_state_and_event(repository) -> None:
    node = checkpointed_node("planner", handler_returning({"plan": {"objective": "x"}, "next_stage": "retriever"}), repository)
    update = node(initial_state("run", "question", ["doc"]))
    stored = repository.get_state("run")
    assert stored["plan"]["objective"] == "x"
    assert stored["last_completed_stage"] == "planner"
    assert [event.event_type for event in repository.list_events("run", 0)][-2:] == ["started", "completed"]


def test_failed_stage_keeps_previous_state(repository) -> None:
    before = repository.get_state("run")
    node = checkpointed_node("researcher", raising_handler(), repository)
    with pytest.raises(WorkflowError):
        node(before)
    assert repository.get_state("run") == before
    assert repository.get("run").status == "failed"
```

- [ ] **Step 2: Implement checkpoint wrapper**

```python
# app/workflow/checkpoints.py
def checkpointed_node(stage: str, handler, repository):
    def run(state: WorkflowState) -> dict:
        repository.start_stage(state["run_id"], dict(state), stage)
        try:
            updates = handler(state)
            merged = merge_state(state, updates)
            merged["last_completed_stage"] = stage
            repository.complete_stage(
                run_id=state["run_id"],
                state=merged,
                current_stage=merged["next_stage"],
                last_completed_stage=stage,
                iteration=int(merged.get("iteration", 0)),
                provider_metrics=list(merged.get("provider_metrics", [])),
                event_payload={},
                report=ResearchReport.model_validate(merged["report"]) if stage == "citation_validator" else None,
            )
            return {**updates, "last_completed_stage": stage}
        except Exception as exc:
            repository.fail_stage(
                state["run_id"], dict(state), stage,
                "workflow_stage_failed", type(exc).__name__,
            )
            raise
    return run
```

The repository methods perform snapshot/event/report and failure/event mutations in single transactions. When `current_stage == "completed"`, `complete_stage()` also marks the run completed, sets `completed_at` and `evidence_sufficient` from the report, and appends a terminal `finished` event carrying `{"status": "completed", "evidence_sufficient": ...}` in that same transaction. The wrapper emits structured stage-start, stage-complete, and stage-failure logs with run ID, stage, safe provider metrics, elapsed time, and exception type only. Configure `WorkflowDependencies.node_wrapper` as `lambda stage, handler: checkpointed_node(stage, handler, runs)` before calling `build_research_graph()`; a graph built for production must never register raw Agent nodes.

- [ ] **Step 3: Write ResearchService validation and lifecycle tests**

```python
# tests/services/test_research_service.py
def test_create_run_requires_ready_documents(service, document_repository) -> None:
    document_repository.seed(document(status="failed"))
    with pytest.raises(WorkflowError, match="document_not_ready"):
        service.create_run("question", ["doc1"])


def test_execute_is_idempotent_after_completion(service) -> None:
    run = service.create_run("question", ["doc1"])
    first = service.execute(run.id)
    second = service.execute(run.id)
    assert second == first
    assert service.graph_invocations == 1
```

Add blank question, missing document, empty selection, failure, report retrieval, event cursor, incomplete-run tests, and an exhausted-Critic test asserting `status == "completed"`, `evidence_sufficient is False`, a persisted insufficiency limitation, and one terminal `finished` event.

- [ ] **Step 4: Implement the service contract**

```python
# app/services/research.py
class ResearchService:
    def __init__(self, document_repository, run_repository, graph, provider_name: str, model: str, recursion_limit: int = 24) -> None:
        self.documents = document_repository
        self.runs = run_repository
        self.graph = graph
        self.provider_name = provider_name
        self.model = model
        self.recursion_limit = recursion_limit

    def create_run(self, question: str, document_ids: list[str]) -> ResearchRun:
        cleaned = question.strip()
        if not cleaned:
            raise WorkflowError("empty_question", "question must not be blank")
        documents = [self.documents.get_document(value) for value in document_ids]
        if not documents or any(document is None for document in documents):
            raise WorkflowError("document_not_found", "one or more documents do not exist")
        if any(document.status != "ready" for document in documents):
            raise WorkflowError("document_not_ready", "all documents must be ready")
        run = ResearchRun.new(cleaned, document_ids, self.provider_name, self.model)
        self.runs.create(run, initial_state(run.id, cleaned, document_ids))
        return run

    def execute(self, run_id: str) -> ResearchReport:
        run = self.runs.get(run_id)
        if run is None:
            raise WorkflowError("run_not_found", "research run does not exist")
        if run.status == "failed":
            raise WorkflowError("run_failed", "failed runs are not retried automatically")
        existing = self.runs.get_report(run_id)
        if existing is not None:
            return existing
        state = self.runs.get_state(run_id)
        try:
            self.graph.invoke(state, {"recursion_limit": self.recursion_limit})
        except Exception as exc:
            current = self.runs.get(run_id)
            if current is not None and current.status != "failed":
                failed_state = self.runs.get_state(run_id)
                self.runs.fail_stage(
                    run_id, failed_state, failed_state["next_stage"],
                    "workflow_failed", type(exc).__name__,
                )
            raise
        report = self.runs.get_report(run_id)
        if report is None:
            raise WorkflowError("report_not_persisted", "workflow completed without a persisted report")
        return report

    def get_run(self, run_id: str) -> ResearchRun:
        run = self.runs.get(run_id)
        if run is None:
            raise WorkflowError("run_not_found", "research run does not exist")
        return run

    def get_report(self, run_id: str) -> ResearchReport | None:
        self.get_run(run_id)
        return self.runs.get_report(run_id)

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]:
        self.get_run(run_id)
        return self.runs.list_events(run_id, after_sequence)

    def list_incomplete_ids(self) -> list[str]:
        return [run.id for run in self.runs.list_incomplete()]

```

- [ ] **Step 5: Write and run restart recovery integration test**

```python
# tests/integration/test_workflow_recovery.py
def test_restart_resumes_running_checkpoint_without_repeating_completed_nodes(app_factory, tmp_path) -> None:
    first = app_factory(tmp_path)
    run = first.research.create_run("question", [first.ready_document.id])
    first.run_repository.seed_running_checkpoint(
        run.id,
        state_after_retriever(run.id, first.ready_document.id),
        current_stage="researcher",
    )

    second = app_factory(tmp_path)
    second.research.execute(run.id)
    assert second.calls["planner"] == 0
    assert second.calls["retriever"] == 0
    assert second.calls["researcher"] == 1
    assert second.research.get_report(run.id) is not None
```

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/workflow/test_checkpoints.py tests/services/test_research_service.py tests/integration/test_workflow_recovery.py -q
```

Expected: PASS.

- [ ] **Step 6: Run the phase gate and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/agents tests/workflow tests/services tests/integration/test_workflow_recovery.py -q`

Expected: PASS without network access.

```powershell
git add app/workflow/checkpoints.py app/services/research.py tests/workflow/test_checkpoints.py tests/services/test_research_service.py tests/integration/test_workflow_recovery.py
git commit -m "feat: persist and resume research workflows"
```
