# Final-rule development regression, recorded before execution

Run the same 20 development questions across four workflows, once each: **80
units**, in a fresh `artifacts/evaluation/real-v1-dev-v8/` directory and dedicated
ledger. This is a development regression after tuning, not held-out evidence.
The [v6 run](full-dev/README.md) and both small engineering confirmations remain
separate. No held-out answer is generated or judged in this stage.

## Generation

- Chat: `qwen3.7-flash-2026-07-15`, temperature 0, thinking off, native strict
  JSON schema, prompt `evaluation-v8`, 4096 output tokens, timeout 60 seconds,
  two provider retries. A truncated answer is a retained execution failure.
- Embeddings: DashScope `text-embedding-v4`, 1024 dimensions. All RAG workflows
  search the complete same eight-document development pool. Chunk 1200, overlap
  200, top-k 5, candidate-k 20, RRF-k 60, minimum similarity 0.15, at most four
  query expansions, two critic loops and 24 workflow steps.
- Shared exact inline citation syntax and known-ID validation apply to all
  grounded answer workflows. Researcher and answer/writer may each make at most
  one explicit citation correction; every physical request remains accounted.
  Request-specific evidence ID enums contain only retrieved context, never gold
  answers. Plain-language insufficiency warnings do not copy internal criteria.
- Serial generation; shuffled case blocks and rotated variant positions; seed
  `20261018`. Preserve exact execution order, clean generation commit and source,
  corpus, dataset, reference-review and configuration identities in the manifest.
- Construct the index once. Report setup latency and embedding requests
  separately. Do not reuse generated answers or query embeddings across units.
- Do not change tracked files while generation or judging runs. Resume only the
  identical manifest; completed failures are not silently regenerated.

## Scoring and review

Use the existing `dev-block-v2` blinded DeepSeek CLI judge and unchanged
`dev-scoring-v2.json` overlay. Score every required key point and every
deterministic answer block. Run up to three judge requests concurrently, with
8192 output tokens and at most one explicit format correction. Retain original
outputs and unresolved judge errors. All reviews are AI reviews; human count
remains zero. No account-mode price is substituted with zero.

Codex source-checks all successful answers for the five prespecified question
IDs `REAL-DEV-001`, `007`, `013`, `019`, `020` (up to 20 answers), every
generation/judge error, and additional questionable assertions if found.
Publish explicit trace-bound accept/correct decisions separately from original
judgments. Correct semantic paraphrases and documented equivalent solutions
receive credit without requiring incidental reference wording. Citation support
requires entailment by the actual attached passages, beyond ID validity.

Publish all 80 units, success denominators, judge missingness, conditional
semantic metrics, citation structure checks, successful-answer latency and
physical request/token/cost records in `docs/evaluation/final-dev/`. Never pool
these observations with v6 as independent repeats. Any comparison is descriptive
development evidence, with shared questions and prior tuning disclosed.

## Freeze decision

Keep the 4096 cap and bounded retrieval/repair rules fixed throughout this run.
An honest remaining truncation or insufficient-evidence failure does not by
itself justify tuning until everything passes. Before a held-out run, audit new
failures, confirm citation/accounting contracts, finish source adjudication and
record the final protocol and identities. A newly discovered implementation
bug requires an explicit repair and separate confirmation, rather than rewriting
this run. Quality limitations remain reportable; they do not authorize selecting
better-looking answers. No formal test result or human-calibrated judge claim
is established by this regression.

## Commands

```powershell
.venv/Scripts/python.exe -m app.evaluation.cli --provider dashscope --embedding-provider dashscope --chat-model qwen3.7-flash-2026-07-15 --real-manifest benchmarks/real-v1/corpus-manifest.json --split dev --allow-ai-reviewed --limit 20 --order-seed 20261018 --structured-output json_schema --max-output-tokens 4096 --capture-responses --ledger artifacts/evaluation/real-v1-dev-v8/ledger.sqlite --output-dir artifacts/evaluation/real-v1-dev-v8/generation
.venv/Scripts/python.exe scripts/judge_development.py --run-dir artifacts/evaluation/real-v1-dev-v8/generation --ledger artifacts/evaluation/real-v1-dev-v8/ledger.sqlite --output-dir artifacts/evaluation/real-v1-dev-v8/judgments --execute
```
