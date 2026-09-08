# Upstream Reuse Audit

## Scope and provenance

This independent, non-commercial project selectively adapts
`NirDiamant/GenAI_Agents` at commit
`4c95ae14cc2462c442b5c064cccd74430d02bc46`. The exact upstream custom license
is retained in the repository root and in
`THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt`. The locally available upstream
history identifies Nir Diamant as the contributor for the pinned snapshot;
no additional notebook-specific contributor is asserted where the available
history does not establish one. This project is not affiliated with or
endorsed by Nir Diamant.

Implemented adapted files include a header with the source notebook, source
cells, pinned commit, changes, and license pointer. A file using only an idea
rather than copied or adapted expression is recorded as `concept-only` below
and in the notices. Existing concept-only headers are retained. This final audit
supersedes target filenames proposed in earlier implementation plans.

## `scientific_paper_agent_langgraph.ipynb` — adapted

- **Source cells:** 13, 15, 19, 21; first available 2024-11-17.
- **Retained code or design:** typed graph state, structured Planner/Critic
  decisions, explicit nodes, conditional routing, and a bounded review loop.
- **Removed defects or unsafe behavior:** mandatory `CORE_API_KEY` access,
  remote CORE API and PDF fetching, direct environment reads in workflow
  nodes, and opaque tool-driven behavior. Local routing has deterministic
  state writes and an explicit retry bound.
- **Adapted local files:** `app/agents/planner.py` and `app/workflow/graph.py`.
  The separately implemented `app/agents/critic.py` and
  `app/workflow/routing.py` support the same bounded design; they do not carry
  adapted-code headers and are not presented as extracted notebook code.
- **Known upstream contributor:** Nir Diamant (pinned upstream history).

## `document_intake_agent_langgraph.ipynb` — adapted

- **Source cells:** 13, 15, 19, 23; first available 2026-07-03.
- **Retained code or design:** a small typed intake state, extension-based
  routing, local-text reading, refusal of unsupported formats, and a
  conditional intake graph.
- **Removed defects or unsafe behavior:** interactive secret prompts, live
  Hushvert format discovery and remote conversion/upload, arbitrary file-path
  reading, and unrestricted server-advertised formats. Local intake accepts
  uploaded bytes only and validates a strict MIME/extension allowlist.
- **Adapted local file:** `app/retrieval/loaders.py`.
  `app/storage/files.py` supplies the project's independently implemented local
  byte-store and path restrictions rather than notebook conversion logic.
- **Known upstream contributor:** Nir Diamant (pinned upstream history).

## `EU_Green_Compliance_FAQ_Bot.ipynb` — concept-only

- **Source cells:** 21, 25, 36; first available 2024-11-17.
- **Retained design:** document chunking, candidate retrieval, query
  expansion/rephrasing, ranking, relevance filtering, and answer fusion.
- **Removed defects or unsafe behavior:** embedded placeholder API-key setup,
  notebook-specific cloud paths, source-loss across retrieval/fusion,
  ambiguous distance-score interpretation, duplicate handling gaps, and
  incompatible agent/vector-store interfaces. Local code preserves source
  metadata and uses deterministic score semantics.
- **Concept-only local files:** `app/retrieval/retriever.py`,
  `app/retrieval/fusion.py`, and `app/agents/retriever.py`, as their headers state.
  The local chunker and NumPy index are original project implementations.
- **Known upstream contributor:** Nir Diamant (pinned upstream history).

## `multi_agent_collaboration_system.ipynb` — concept-only

- **Source cells:** 6, 11–21; first available 2024-09-09.
- **Retained design:** role separation, sequential hand-off, shared
  context accumulation, and final synthesis. No copied sequential baseline is
  shipped; the implemented benchmark compares direct chat, RAG, planned RAG,
  and the product graph.
- **Removed defects or unsafe behavior:** module-global LLM coupling,
  mutable positional context, console-only progress reporting, and direct
  free-form hand-offs without typed state or citation checks.
- **Concept-only local files:** `app/agents/researcher.py`,
  `app/agents/writer.py`, and `app/evaluation/workflows.py`. Researcher carries a
  concept-only header; Writer and evaluation adapters implement project-specific
  contracts. There is no `app/evaluation/baselines.py` in the delivered tree.
- **Known upstream contributor:** Nir Diamant (pinned upstream history).

## `trace_based_agent_evaluation.ipynb` — adapted

- **Source cells:** 5, 9, 13, 15, 19; first available 2026-08-28.
- **Retained code or design:** typed evaluation traces, per-trace scoring,
  nearest-rank percentile latency, exception isolation,
  suite aggregation, and structured quality-gate failures.
- **Removed defects or unsafe behavior:** tutorial literals coupled to weather
  and order tools, unredacted arbitrary trace payload assumptions, and a
  single in-notebook assertion. Local evaluation uses project-specific,
  redacted trace fields, versioned fixtures, and pytest quality gates.
- **Adapted local files:** `app/evaluation/trace.py` and
  `app/evaluation/metrics.py`. Their headers identify source cells, commit,
  changes, and license. Project-specific orchestration and thresholds live in
  `app/evaluation/runner.py` and `app/evaluation/quality_gate.py`; behavioral
  tests are in `tests/evaluation/`.
- **Known upstream contributor:** Nir Diamant (pinned upstream history).
