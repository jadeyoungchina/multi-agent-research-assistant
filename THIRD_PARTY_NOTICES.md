# Third-Party Notices

This independent, non-commercial project adapts selected material from
[NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) at
commit `4c95ae14cc2462c442b5c064cccd74430d02bc46`. It is not affiliated with,
or endorsed by, Nir Diamant. The upstream custom non-commercial license is
vendored verbatim in [LICENSE](LICENSE) and
`THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt`.

| Upstream file | Source cells | First available | Local targets | Treatment |
|---|---:|---|---|---|
| `scientific_paper_agent_langgraph.ipynb` | 13, 15, 19, 21 | 2024-11-17 | `app/agents/planner.py`, `app/workflow/graph.py` | adapted |
| `document_intake_agent_langgraph.ipynb` | 13, 15, 19, 23 | 2026-07-03 | `app/retrieval/loaders.py` | adapted |
| `EU_Green_Compliance_FAQ_Bot.ipynb` | 21, 25, 36 | 2024-11-17 | `app/retrieval/retriever.py`, `app/retrieval/fusion.py`, `app/agents/retriever.py` | concept-only |
| `multi_agent_collaboration_system.ipynb` | 6, 11–21 | 2024-09-09 | `app/agents/researcher.py`, `app/agents/writer.py`, `app/evaluation/workflows.py` | concept-only |
| `trace_based_agent_evaluation.ipynb` | 5, 9, 13, 15, 19 | 2026-08-28 | `app/evaluation/trace.py`, `app/evaluation/metrics.py` | adapted |

Every adapted target carries its source notebook, source cells, pinned commit,
changes, and license pointer in its file header. Detailed retained material,
removals, target files, contributors, and treatment decisions are maintained
in `docs/upstream-analysis.md`.

The table describes the implemented files, superseding planned mappings in
historical implementation plans. Concept-only rows identify design influence,
not copied notebook expression. The repository verifier checks each explicitly
`adapted` target against the pinned commit. The known upstream contributor for
the pinned snapshot is Nir Diamant; the audit does not assert additional
notebook-specific authors without evidence.

The `benchmarks/real-v1/raw` directory additionally contains 24 unmodified
public documentation snapshots from Pydantic (`v1.10.17`, `v2.8.2`) and
FastAPI (`0.112.2`). Their MIT notices are preserved in
`benchmarks/real-v1/licenses/`. Exact upstream commits, per-file URLs,
retrieval timestamps, hashes, license associations, and transformations are
recorded in `benchmarks/real-v1/corpus-manifest.json`. Normalized copies only
add provenance headers and normalize line endings; no upstream endorsement
is implied. Dataset questions are separately AI authored and human review
remains pending; see `benchmarks/real-v1/README.md`.
