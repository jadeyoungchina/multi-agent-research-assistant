# Technical report: implementation and real-document evaluation

The delivered application turns local documents and a question into a persisted
research report with inspectable evidence. The design favors explicit state,
bounded revision, and reproducible software checks. It is an independent
non-commercial learning project derived in part from the pinned upstream
notebooks documented in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Engineering decisions

FastAPI and native browser assets keep deployment within one Python process.
Pydantic validates configuration, provider schemas, workflow models, and API
payloads. The same service graph powers both the web application and the
`multi_agent_rag` evaluation adapter, so the benchmark exercises product behavior.

SQLite provides local transactional persistence; NumPy cosine retrieval avoids
requiring a remote vector database. Evidence IDs survive query fusion, synthesis,
writing, and citation rendering. The tradeoff is an in-memory scan of selected
document vectors on each search, appropriate for small local corpora rather than
large-scale retrieval. Chat and embeddings have separate provider settings.

The Critic combines model judgments with deterministic evidence-ID checks and
completion-criteria matching. These checks catch structural inconsistencies;
they do not establish that a passage entails a claim. A bounded revision loop
and graph recursion limit prevent endless research. Literal matching is deliberately
simple and may mark a semantically good answer insufficient. The writer retains
insufficiency in report limitations instead of silently claiming completion.

Stage snapshots and append-only ordered events allow restart recovery and SSE
replay. They do not guarantee exactly-once model billing if a process stops
between a provider response and its durable checkpoint. Concurrency is local to
one process; multiple application workers need additional coordination.

## Verification evidence and interpretation

The executable evidence is the test suite and generated benchmark output:

```bash
python -m pytest -m "not live" -q
python scripts/verify_repository.py
python -m app.evaluation.cli --provider fake --variants all --output-dir artifacts/evaluation/fake --enforce-gate
```

Tests cover provider failures/repair, parsing and path restrictions, retrieval,
state/routing/citations, stage recovery, storage, API/SSE, browser behavior,
product smoke, evaluator metrics, and release hygiene. Browser behavior tests
use Node.js 22, also configured by CI. CI runs Python 3.11 with
fake providers and blank keys, and uploads generated fake benchmark evidence.
Dependency installation downloads packages; CI makes no live model calls.

Benchmark thresholds are declared acceptance criteria, and the 30 cases are a
synthetic regression fixture rather than a representative research population.
Fake results check software behavior and cannot establish real-model answer
quality. [Experiments](experiments.md) separates those metrics from the real study.

## Real-document experiment

The separate [held-out packet](evaluation/heldout-test/RESULTS.md) uses pinned
Pydantic/FastAPI documentation, AI-authored questions and trace-bound AI scoring.
Twenty development questions supported calibration; 40 questions from eight other
document families form the held-out comparison. All three RAG workflows search
the same 16-document test candidate pool, with answer keys kept outside workflow
inputs. A 257-input freeze binds implementation, configuration, reference data,
three serial execution orders, the review sample and statistics before generation.

The generation model is `qwen3.7-flash-2026-07-15`; embeddings use
`text-embedding-v4`. The 40 x 4 x 3 matrix records **480 executions: 473 successful
and seven failed**. Primary quality averages three repeats within each question,
including executed-failure zeros, before aggregating 40 question means.

| Workflow | Primary quality | Successful / 120 | Median successful seconds | Physical answer attempts |
| --- | ---: | ---: | ---: | ---: |
| Direct LLM | 0.643 | 120 | 5.961 | 120 |
| Plain RAG | 0.919 | 119 | 3.973 | 248 |
| Single-agent RAG | 0.821 | 119 | 8.646 | 815 |
| Multi-agent RAG | 0.878 | 115 | 49.848 | 2070 |

The prespecified multi-minus-single correctness difference is **+5.69 percentage
points**, with 40 complete paired questions and exploratory family-bootstrap 95%
interval **[+1.39, +9.72] pp** (10,000 draws; seed 20261023). Plain RAG has the highest
observed quality overall; multi-agent median successful latency is about 5.8 times
single-agent. More orchestration therefore does not produce an overall winner on
this task set. Citation structure validity also differs from semantic support:
multi-agent conditional support is 0.561 over 114 applicable successful answers,
versus 0.856 over 103 for plain RAG. These are descriptive, conditional means.

Original judging and source proposals use the same DeepSeek alias in separate
contexts. The 80 prespecified source checks plus 17 extra error/failure checks
are complete; 37 explicit Codex follow-ups compare problematic judgments against
saved answers and source passages. The original analysis and 15 original judge
errors remain visible alongside corrections. The source-audit procedure followed
the initial aggregate analysis, and correlated reviewer errors remain possible.
Human validation counts are zero; the user excluded subsequent human review from
this delivery. That scope decision does not change the study's limitations.

All **3960 physical requests** are retained across indexing, answer generation,
initial judging and source review. Seventeen requests have unknown usage and
DeepSeek account prices are unverified, so total charges remain null. The paired
incremental answer-cost estimate covers only 30/40 question pairs and excludes
indexing and judging/review. Serial execution on one machine, public documents
possibly seen in pretraining and eight document families limit inference.

The [executed notebook](../output/jupyter-notebook/real-evaluation-analysis.ipynb)
recomputes the public packet without model calls. The [reproduction guide](evaluation/heldout-test/REPRODUCE.md)
documents dependencies and exact-byte checkout settings, while [seven selected
cases](evaluation/heldout-test/FAILURE-CASES.md) explain unsupported claims,
retrieval misses, partial answers and unresolved execution failures.

## Contribution boundaries

The [upstream audit](upstream-analysis.md) and [notices](../THIRD_PARTY_NOTICES.md)
identify adapted Planner/graph, intake and trace/metric material separately from
conceptual influence. Project-specific engineering includes local storage and
recovery, provider accounting, API/SSE integration, the browser interface,
source-preserving retrieval and the frozen evaluation/publication workflow.
Development and evaluation used Codex and DeepSeek assistance. These artifacts
document implementation and evidence. The contribution map follows file-level
provenance and separates project-specific engineering from notebook adaptations.

## Demonstration and presentation

Run `python examples/run_demo.py` after installing the project dependencies for
an isolated, preloaded fake-provider demonstration. Its explicit settings ignore
the existing environment and use a fresh local database. `--check` exercises
actual API, workflow, SSE, citation and report-reopen behavior. The optional
browser helper verifies the real UI and captures source inspection without
relaxing the application's content security policy.

[Project brief](portfolio/project-brief.md), [demo guide](portfolio/demo-guide.md)
and [three-minute walkthrough](portfolio/presentation.zh-CN.md) provide a shorter
review path alongside the implementation and experiment details above.

## Privacy and unresolved product limits

The application has no authentication, multi-tenant isolation, database
encryption, automatic expiry, internet search, or OCR. Use loopback for personal
operation. Real embedding calls send document chunks to a provider; real chat
calls send questions and evidence. Historical reports/snapshots may retain
source text after document deletion. Credential redaction covers structured
sensitive fields and safe error boundaries; arbitrary text should not be
assumed sanitized. Review logs and outputs before sharing.

Citation identity checks do not judge semantic support, contradiction resolution,
or real-world correctness. Browser rendering, MIME/size limits, bounded provider
calls, and repository secret checks reduce specific failure modes but are not
a general security certification. Prompt-injected content remains untrusted
evidence for the language model.

## License and attribution

The root license and vendored copy preserve upstream terms exactly after newline
normalization. [Upstream analysis](upstream-analysis.md) distinguishes expression
adapted in source headers from conceptual influence. Commercial use requires
written permission from the upstream licensor.
