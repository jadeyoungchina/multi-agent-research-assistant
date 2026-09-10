# Multi-Agent Research Assistant

[简体中文](README.zh-CN.md)

A local document research application with a FastAPI API, a same-origin web UI,
and a LangGraph workflow: Planner → Retriever → Researcher → Critic → Writer →
Citation Validator. Upload PDF, Markdown, or UTF-8 TXT files, ask a question,
follow progress, and inspect the sources attached to the report.

This repository was reconstructed on 2026-09-08 after the original local project was accidentally deleted. Commit dates restore the documented development milestones; they are not the original Git objects or an unreconstructed historical record.

This is an independent, non-commercial learning project that selectively adapts
[NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) at commit
`4c95ae14cc2462c442b5c064cccd74430d02bc46`. It is not affiliated with or endorsed
by Nir Diamant. The upstream custom [LICENSE](LICENSE) restricts commercial use;
commercial permission must be obtained from the licensor in writing. Attribution,
source cells, and file mappings are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Features and boundaries

- Local parsing, source-preserving chunks, embeddings, cosine search, and reciprocal rank fusion.
- Structured agent outputs, bounded Critic revision, citation ID validation, and explicit evidence insufficiency.
- SQLite documents, vectors, reports, stage checkpoints, and ordered events; incomplete runs are rescheduled on startup.
- Upload/selection UI, report rendering, source excerpts, SSE progress, and polling fallback.
- DashScope and OpenAI compatible providers; deterministic fake chat and embeddings for offline demonstration and CI.
- A synthetic [30-case benchmark](benchmarks/cases.jsonl), four workflow variants, and JSON/CSV/Markdown reports.

The application has no internet search, OCR, authentication, multi-tenant access
control, or distributed queue. Citation validation checks source identity and
format; it does not establish that a claim is true. Fake output demonstrates
the software workflow and is not evidence of real-model research quality.

## Quick start: offline fake providers

Use Python 3.11 (the supported range is 3.11–3.12). Installation needs package
downloads; the fake application and tests make no model-provider calls.
Run commands from the repository root:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or
`.venv\Scripts\Activate.ps1` in PowerShell, then:

```bash
python -m pip install -r requirements/upstream-requirements.txt
python -m pip install -r requirements/project-requirements.txt
```

In Bash:

```bash
CHAT_PROVIDER=fake EMBEDDING_PROVIDER=fake RETRIEVAL_MIN_SIMILARITY=-1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In PowerShell:

```powershell
$env:CHAT_PROVIDER = 'fake'
$env:EMBEDDING_PROVIDER = 'fake'
$env:RETRIEVAL_MIN_SIMILARITY = '-1'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [the local UI](http://127.0.0.1:8000), upload files from
`examples/demo-corpus`, select them, and ask “What are the project's goals and
evaluation limitations?” API documentation is at [local Swagger UI](http://127.0.0.1:8000/docs).
The fake provider synthesizes supplied evidence deterministically; it is not an LLM.
The demo's -1 similarity threshold keeps evidence available with toy hash
embeddings. The real-provider default is 0.15; do not use the demo threshold to
infer retrieval quality.

## DashScope / Qwen setup

Copy `.env.example` to `.env`. Set `CHAT_PROVIDER=dashscope` and
`EMBEDDING_PROVIDER=dashscope`, and put your own DashScope credential in the
blank `DASHSCOPE_API_KEY` field of that local file. Keep the following defaults
unless your account uses different supported models or a regional endpoint:

```dotenv
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL=qwen3.7-flash
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_BATCH_SIZE=10
```

The DashScope-compatible defaults keep embedding batches within the current
`text-embedding-v4` limit. Structured JSON requests also disable Qwen thinking
output so workflow stages return bounded machine-readable responses.

Unset earlier shell provider and similarity overrides or use a fresh terminal,
activate the environment, and start the real-provider configuration:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Shell variables override `.env`.
The configured model names do not imply account availability or a successful
live run. Real-provider requests can incur charges and send document text to
the provider. OpenAI can be selected with both provider fields set to `openai`
and a local `OPENAI_API_KEY`; its endpoint/model fields are in `.env.example`.
Use separate database/upload paths when changing embedding models, and ingest
documents again: stored vectors are tied to their embedding model.

## API and SSE examples

These examples use Bash curl syntax; use `curl.exe` with your shell's quoting
rules on Windows. The explicit MIME type is required for upload validation.

```bash
curl -F 'files=@examples/demo-corpus/project-overview.md;type=text/markdown' http://127.0.0.1:8000/api/documents
curl http://127.0.0.1:8000/api/documents
```

Upload returns HTTP 201 with `{"documents":[...]}`. Copy a returned `id` into
`DOCUMENT_ID` below; it is a placeholder, not a fixed demo identifier:

```bash
curl -i -H 'Content-Type: application/json' -d '{"question":"What are the project goals?","document_ids":["DOCUMENT_ID"]}' http://127.0.0.1:8000/api/research
```

HTTP 202 returns `run_id`, `status`, and a `Location` header. Replace `RUN_ID`
below with that value. Run states are `queued`, `running`, `completed`, `failed`.

```bash
curl -N http://127.0.0.1:8000/api/research/RUN_ID/events
curl http://127.0.0.1:8000/api/research/RUN_ID
curl -N -H 'Last-Event-ID: 5' 'http://127.0.0.1:8000/api/research/RUN_ID/events?after=3'
```

The run response contains `run` and `report` (null until available). SSE emits
`event: workflow`, an integer `id`, and JSON with `sequence`, `run_id`, `stage`,
`event_type`, `payload`, and `created_at`. Resume uses the greater of `after`
and `Last-Event-ID`; the example replays events after sequence 5. Keep-alive
comments maintain idle streams; terminal streams close after queued events drain.
Delete a document with `DELETE /api/documents/DOCUMENT_ID`. Health is at `/health`;
provider readiness there is configuration readiness, not a live endpoint probe.

## Tests and experiments

The full test suite also requires Node.js 22 for the browser behavior harness.
The application itself has no Node build step. CI configures both runtimes.

```bash
python -m pytest -m "not live" -q
python scripts/verify_repository.py
python -m app.evaluation.cli --provider fake --variants all --output-dir artifacts/evaluation/fake --enforce-gate
```

The benchmark writes `results.json`, `results.csv`, and `report.md` under the
chosen output directory. The release gate applies to `multi_agent_rag`; the
other three variants are comparisons. Generated results stay untracked and CI
uploads them as artifacts. See [experiments](docs/experiments.md) for exact metric
definitions, thresholds, real-provider commands, and limitations. No real-model
quality or speedup is claimed by the fake benchmark.

## Privacy and development

Uploads default to `data/uploads`; SQLite defaults to `data/research_assistant.db`.
Documents, extracted text, vectors, questions, state snapshots, and reports stay
on your machine in fake mode. With a real embedding/chat provider, document text,
questions, and evidence are transmitted to the configured provider. Local files
are not encrypted by this application. There is no automatic retention policy;
deleting a document does not promise to erase its text from earlier run reports
or snapshots. Treat all local data and custom evaluation outputs as private.
The synthetic governance corpus is fictional and does not define app retention.

Bind to loopback for personal use. Store credentials only in local environment
configuration, never in Git, screenshots, or evaluation reports. The verifier
checks tracked working-tree files for runtime data, common credential patterns,
licenses, benchmark evidence, and adapted-file attribution; it is not a scan of
Git history or a substitute for reviewing staged changes.

Read [architecture](docs/architecture.md), [technical report](docs/technical-report.md),
[reconstruction history](docs/reconstruction-history.md), and the
[release checklist](docs/release-checklist.md) for developer and release details.
