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
