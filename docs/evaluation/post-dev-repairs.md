# Repairs after the fixed v6 development run

The original [80-unit report](full-dev/README.md), including three failures and
ten malformed-citation answers, remains unchanged. These repairs produce prompt
v7 and are engineering follow-up, not a replacement development estimate.

- Exact request-local labels may carry an anchored `Cite:` prefix. Strip this
  formatting only when the resulting label/ID exists in the supplied mapping.
  Unknown labels remain rejected; prose is never matched fuzzily to a source,
  and no citation token is inserted into uncited text.
- Researcher passage text in ID fields is rejected. One optional model correction
  can replace invalid fields with actual supplied labels or remove unsupported
  findings. It still fails after a second invalid response. All original,
  corrected and failed requests retain their usage. The configured default is
  one correction; the run configuration records it separately.
- All three RAG workflows now use the same exact inline-token and declared-ID
  consistency rules. Plain/single-agent RAG can make one bounded formatting
  correction, using the same answer-repair allowance as the multi-agent writer.
  This changes future execution reliability and request cost; historical v6
  success flags are not relabeled. Formatting success does not establish entailment.
- Writer and final renderer share the same plain-language insufficiency rule.
  The full internal critique remains in workflow state; user reports receive
  the model's document limitations and a single generic insufficiency warning.
  End-to-end tests check both Chinese and English questions.

Seven new regression cases first failed on the original code. An additional
case checks that valid inline IDs must also be declared. Existing success fixtures
now contain actual inline tokens, and old tests expecting internal critique text
now expect the public warning. The 4096-token cap remains; truncation is still
an explicit failure mode, not silently retried at a larger limit.

## Targeted live confirmation, selected before execution

Use the fixed Qwen snapshot, real embeddings, strict schema, cap 4096 and seed
20261018. Keep a separate ledger and new directories:

1. Questions 019 and 020, multi-agent only: the observed researcher/writer failures.
2. Question 007, plain and single-agent RAG: check the shared inline citation guard.

These four units confirm repair behavior; they are not independent accuracy
repeats, do not replace v6 failures and do not establish general reliability.
Keep any new failures. No held-out question is used.

The [v7 confirmation](post-dev-confirmation.json) completed 3/4 units successfully
using clean commit `febc470`. Question 019 still returned passage text after the
single model correction. This failure is retained alongside the three successes;
69 physical request receipts include both index constructions and every attempt.

## Request-specific ID constraints, v8

The compatible provider now binds evidence ID arrays to the exact labels supplied
for that request in both the schema instruction and native strict JSON schema.
This applies to the researcher, writer, plain RAG and single-agent RAG equally.
An empty evidence set constrains ID arrays to length zero. Model classes and
other requests are not mutated. Custom/fake providers can use the existing
structured-output interface; local citation checks remain mandatory everywhere.
No gold/reference answer enters this request schema.

Before execution, select question 019, multi-agent only, for one further live
confirmation with the same model, cap, retry/repair limits and seed in a fresh
`enum-confirmation` directory and separate ledger. Preserve its outcome even if
it fails; do not pool it with the 80-unit report or the four-unit v7 confirmation.

The [separate v8 confirmation](enum-confirmation.json) succeeded for question 019
at clean commit `4553429`, retaining the original 4096-token cap and bounded
repair/retrieval limits. This is **1/1 engineering confirmation**, not a rerun
of the full development matrix. All researcher responses used supplied labels,
and final inline citations passed the shared validator. The original v6 and v7
failures remain published. It still used two critic loops and about 75 seconds;
the ID constraint does not establish efficient retrieval or semantic entailment.

Validation: 626 offline tests pass, one live test deselected. The 120-trace fake
release gate, real-document fake pipeline and both original public packets pass.
Recheck the separate confirmations and their physical receipts with:

```powershell
.venv/Scripts/python.exe scripts/verify_dev_confirmation.py
```

Before experiment freeze, repeat development validation with the final common
rules and settle the remaining output-length/retrieval limits. No held-out run
or frozen primary comparison is claimed here.
