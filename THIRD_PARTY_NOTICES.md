# Third-Party Notices

This independent, non-commercial project adapts selected material from
[NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) at
commit `4c95ae14cc2462c442b5c064cccd74430d02bc46`. It is not affiliated with,
or endorsed by, Nir Diamant. The upstream custom non-commercial license is
vendored verbatim in [LICENSE](LICENSE) and
`THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt`.

| Upstream file | Source cells | First available | Local targets | Treatment |
|---|---:|---|---|---|
| `scientific_paper_agent_langgraph.ipynb` | 13, 15, 19, 21 | 2024-11-17 | workflow, Planner, Critic | adapted |
| `document_intake_agent_langgraph.ipynb` | 13, 15, 19, 23 | 2026-07-03 | document routing | adapted |
| `EU_Green_Compliance_FAQ_Bot.ipynb` | 21, 25, 36 | 2024-11-17 | chunk/retrieval/query fusion | adapted |
| `multi_agent_collaboration_system.ipynb` | 6, 11–21 | 2024-09-09 | sequential baseline | adapted/concept-only by file |
| `trace_based_agent_evaluation.ipynb` | 5, 9, 13, 15, 19 | 2026-08-28 | trace evaluator | adapted |

Every adapted target carries its source notebook, source cells, pinned commit,
changes, and license pointer in its file header. Detailed retained material,
removals, target files, contributors, and treatment decisions are maintained
in `docs/upstream-analysis.md`.
