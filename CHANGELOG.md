# 1.1.0 - 2026-10-21

- Added a pinned real-document benchmark, four comparable workflows, provider
  request accounting and resumable frozen experiment execution.
- Published 480 held-out executions, original AI judgments, 97 source/failure
  checks, paired statistics, seven selected failure cases and request receipts.
- Added an executed analysis notebook, three figure sets and offline reproduction
  against the original `7e0cf6c` source snapshot, separate from later product fixes.
- Added an isolated one-command local demo, verified browser screenshots, an
  English project brief, a technical report and a three-minute presentation script.
- Fixed default application startup when configuration is loaded internally.
- Brought results, source inspection and evidence links to the top of both READMEs;
  synchronized package and API versions for this release.

Plain RAG scores highest in this small held-out study. Multi-agent RAG improves
on the single-agent comparison while taking substantially longer. AI-assisted
references and scoring have no independent human validation. See the
[full results](docs/evaluation/heldout-test/RESULTS.md) and
[reproduction guide](docs/evaluation/heldout-test/REPRODUCE.md).
The upstream non-commercial license and attribution remain in force.

# 1.0.0 - 2026-09-08

- Added typed providers, deterministic offline fake chat/embeddings, and local SQLite storage.
- Added PDF/Markdown/TXT ingestion, source-preserving retrieval, and query fusion.
- Added the five-role LangGraph workflow, bounded critique, citation validation, and stage recovery.
- Added FastAPI upload/research/SSE APIs and the same-origin web interface.
- Added the synthetic benchmark, four workflow adapters, metrics, reports, and fake release gate.
- Added bilingual documentation, offline CI, repository hygiene checks, and pinned upstream attribution.

The upstream custom non-commercial license remains in force. This release does
not claim measured real-model quality or comparative performance; see
[experiments](docs/experiments.md) and [release checklist](docs/release-checklist.md).
