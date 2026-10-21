# Multi-Agent Research Assistant

A non-commercial engineering project investigating when structured multi-agent
orchestration helps local-document research, and what it costs in latency,
reliability and citation support.

## Project summary

The application accepts local PDF, Markdown and text files, preserves their source
locations during chunking and retrieval, and produces reports with inspectable
citations. A FastAPI API and browser interface expose upload, progress and evidence
inspection. A bounded LangGraph workflow separates planning, retrieval, synthesis,
critique and writing; SQLite stores stage snapshots, ordered events and reports.
Provider interfaces support real Qwen/OpenAI-compatible services and deterministic
offline demonstrations.

The evaluation compares direct generation, plain RAG, planned single-agent RAG and
the multi-agent workflow. A frozen 40-question test across eight documentation
families, repeated three times, records 480 executions, including seven failures.
Plain RAG obtains the highest observed primary quality, 0.919. Multi-agent RAG
scores 0.878 versus 0.821 for single-agent RAG, but its median successful latency
is about 5.8 times greater. The prespecified paired difference is +5.69 percentage
points, with an exploratory family-bootstrap 95% interval of [+1.39, +9.72] pp.

The main lesson is an engineering tradeoff: extra agents can improve a specific
comparison while a simpler retrieval baseline performs better overall. Public
traces, original judgments, explicit source corrections, per-request usage and an
offline analysis notebook make that conclusion inspectable.

## Implementation and evidence

| Engineering area | Implemented behavior | Review entry point |
| --- | --- | --- |
| Evidence retrieval | Source-preserving chunks, model-tagged vectors, cosine search and query fusion | [Architecture](../architecture.md), [retrieval code](../../app/retrieval) |
| Bounded execution | Typed stages, critique limits, graph routing and persisted checkpoints | [Workflow code](../../app/workflow), [research service](../../app/services/research.py) |
| Application integration | Upload API, background execution, SSE replay and citation inspection | [API](../../app/api), [browser](../../app/static) |
| Evaluation design | Shared candidate pools, hidden answer keys, three repeats and question-level paired analysis | [Frozen protocol](../evaluation/frozen-experiment.md) |
| Failure/accounting discipline | Failed executions, initial judge errors, unknown usage and corrections retained | [Results](../evaluation/heldout-test/RESULTS.md), [failure cases](../evaluation/heldout-test/FAILURE-CASES.md) |
| Reproducible analysis | Verified public packet, executed notebook, tables and three figure sets | [Reproduce](../evaluation/heldout-test/REPRODUCE.md) |

## Attribution, tools and limits

The project selectively adapts `NirDiamant/GenAI_Agents` under its custom
non-commercial license; it does not claim to invent the multi-agent pattern.
[Exact adapted files and source cells](../../THIRD_PARTY_NOTICES.md) are documented.
Implementation and evaluation used Codex and DeepSeek assistance, including
AI-authored reference material, AI scoring and explicit AI source adjudications.
Independent human validation was not performed and is outside the current
delivery scope. The small public-document benchmark does not establish general
architectural superiority, and incomplete billing information is kept unknown.

The application is a local single-user system without authentication, OCR or
internet search. Its offline fake mode demonstrates software behavior; model
quality claims come only from the separately recorded real-model experiment.

[Technical report](../technical-report.md) · [Source repository](../../README.md)
