# Technical report: v1.0.0 reconstruction

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

The Critic's deterministic checks prevent unsupported evidence references from
being accepted as sufficient. A bounded revision loop and graph recursion limit
prevent endless research. Literal matching of completion criteria is deliberately
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

No fixed pass count or performance table is embedded here: use the current
command output, CI logs, and generated `results.json`/`report.md` for a specific
revision. Benchmark thresholds are declared acceptance criteria, and the 30
cases are a synthetic fixture rather than a representative research population.
The four variants share inputs but have different orchestration. Their fake
results cannot support claims that multiple agents improve real-model quality.
[Experiments](experiments.md) documents scoring, source-filtered recall, lexical
answer metrics, cost exclusions, and the limits of those measurements.

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

## Reconstruction and license

The original local project was accidentally deleted. The reconstructed dates
restore documented milestones rather than original Git objects; the exact
disclosure and timeline are in [reconstruction history](reconstruction-history.md).
The root license and vendored copy preserve upstream terms exactly after newline
normalization. [Upstream analysis](upstream-analysis.md) distinguishes expression
adapted in source headers from conceptual influence. Commercial use requires
written permission from the upstream licensor.
