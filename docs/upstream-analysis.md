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

Planned adapted files must include a header with the source notebook, source
cells, pinned commit, changes, and license pointer. A file using only an idea
rather than copied or adapted expression is marked `concept-only` in that
file's header and below.

## `scientific_paper_agent_langgraph.ipynb` — adapted

- **Source cells:** 13, 15, 19, 21; first available 2024-11-17.
- **Retained code or design:** typed graph state, structured Planner/Critic
  decisions, explicit nodes, conditional routing, and a bounded review loop.
- **Removed defects or unsafe behavior:** mandatory `CORE_API_KEY` access,
  remote CORE API and PDF fetching, direct environment reads in workflow
  nodes, and opaque tool-driven behavior. Local routing has deterministic
  state writes and an explicit retry bound.
- **Receiving local files:** `app/agents/planner.py`, `app/agents/critic.py`,
  `app/workflow/graph.py`, and `app/workflow/routing.py`.
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
- **Receiving local files:** `app/retrieval/loaders.py` and
  `app/storage/files.py`.
- **Known upstream contributor:** Nir Diamant (pinned upstream history).

## `EU_Green_Compliance_FAQ_Bot.ipynb` — adapted

- **Source cells:** 21, 25, 36; first available 2024-11-17.
- **Retained code or design:** document chunking, candidate retrieval, query
  expansion/rephrasing, ranking, relevance filtering, and answer fusion.
- **Removed defects or unsafe behavior:** embedded placeholder API-key setup,
  notebook-specific cloud paths, source-loss across retrieval/fusion,
  ambiguous distance-score interpretation, duplicate handling gaps, and
  incompatible agent/vector-store interfaces. Local code preserves source
  metadata and uses deterministic score semantics.
- **Receiving local files:** `app/retrieval/chunking.py`,
  `app/retrieval/index.py`, and `app/retrieval/retriever.py`.
- **Known upstream contributor:** Nir Diamant (pinned upstream history).

## `multi_agent_collaboration_system.ipynb` — adapted/concept-only by file

- **Source cells:** 6, 11–21; first available 2024-09-09.
- **Retained code or design:** role separation, sequential hand-off, shared
  context accumulation, and final synthesis as an experimental baseline.
- **Removed defects or unsafe behavior:** module-global LLM coupling,
  mutable positional context, console-only progress reporting, and direct
  free-form hand-offs without typed state or citation checks.
- **Receiving local files:** `app/evaluation/baselines.py` is adapted for the
  sequential baseline; `app/agents/researcher.py` and `app/agents/writer.py`
  are `concept-only` role-separation uses.
- **Known upstream contributor:** Nir Diamant (pinned upstream history).

## `trace_based_agent_evaluation.ipynb` — adapted

- **Source cells:** 5, 9, 13, 15, 19; first available 2026-08-28.
- **Retained code or design:** immutable evaluation cases/traces, weighted
  per-trace scoring, nearest-rank percentile latency, exception isolation,
  suite aggregation, and structured quality-gate failures.
- **Removed defects or unsafe behavior:** tutorial literals coupled to weather
  and order tools, unredacted arbitrary trace payload assumptions, and a
  single in-notebook assertion. Local evaluation uses project-specific,
  redacted trace fields, versioned fixtures, and pytest quality gates.
- **Receiving local files:** `app/evaluation/models.py`,
  `app/evaluation/evaluator.py`, `app/evaluation/gates.py`, and
  `tests/evaluation/`.
- **Known upstream contributor:** Nir Diamant (pinned upstream history).
