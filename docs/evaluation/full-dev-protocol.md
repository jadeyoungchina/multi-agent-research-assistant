# Full development calibration, recorded before execution

This follows the five-question development pilot. It covers the existing 20
development questions across all four workflows, once each: **80 units**. Five
questions have already been used for tuning; this is not an independent test.
No held-out question is generated, judged or used for prompt tuning here.

## Fixed generation configuration

- All chat roles: `qwen3.7-flash-2026-07-15`, temperature 0, thinking disabled,
  strict JSON schema, 4096 output tokens, two provider retries, prompt v6.
- DashScope `text-embedding-v4`, 1024 dimensions; chunk size 1200, overlap 200,
  top-k 5. Every RAG variant searches the complete same eight-document dev pool.
- At most two critic loops and one writer citation repair. Unknown evidence
  IDs still fail validation. A bounded repair is an additional paid attempt.
- Serial generation, shuffled question blocks and rotated workflow positions,
  seed `20261018`. Twenty blocks balance each workflow across each position
  five times. Exact execution order and all source/configuration hashes are saved.
- Index construction is shared and reported separately from answer latency
  and per-workflow request costs. Dedicated generation/judge ledger; the separate
  preparation ledger contains the reference clarification review.

The writer now receives an evidence-sufficiency flag rather than internal critic
reasoning. A deterministic plain-language warning preserves insufficiency
without copying agent names, completion criteria or revision bookkeeping.
Other agent behavior, retrieval limits and model settings remain as in the pilot.
No stopping rule uses gold labels or expected insufficiency.

Before the full run, two saved pilot answers exercised the new judge through
the real CLI. Both passed structural validation. Source inspection found one
pure missing-evidence summary mislabeled as factual because it carried a
citation. The rubric now explicitly explains that citations do not turn scoped
limitations into domain facts, including in mixed blocks. Original calibration
responses and this change precede full development generation; their requests
are preparation costs, separate from the full run.

## Scoring, fixed before generation

The original `dev.jsonl`, its AI reference audit, and historical pilot scores
remain unchanged. `dev-scoring-v2.json` applies two explicit **scoring-only**
clarifications: question 007 accepts documented equivalent validation input
methods, and 019 does not require an incidental import alias or detail-field
aside. The original DeepSeek review, concerns, local Pydantic 2.8.2 check and
Codex disposition are in `dev-scoring-review-v2.json`. This overlay cannot be
used for held-out cases and is never supplied to generation workflows.

DeepSeek Flash via the official CLI scores all successful answers in fresh,
blinded contexts (no workflow/model/cost/timing labels). It scores every required
key point 0/1/2 and every deterministically selected Markdown answer block.
Block indices avoid answer-text and long-ID transcription. All blocks, including
uncited assertions and code, must be audited; omissions/duplicates are errors.
Citation support is full entailment of **all factual assertions in that block**
by its attached exact evidence. Missing citations remain uncited regardless of
factual correctness. Headings and scoped missing-evidence limitations are N/A.

Citation metrics now use **answer blocks**, not the pilot's model-selected atomic
claims. These rates must not be directly compared. Original retrieval recall
remains a diagnostic over original gold spans, which may prescribe one of several
valid solutions; it is not semantic answer correctness.

The judge may make one explicit format correction with every raw response and
physical attempt retained. Transport failures are not automatically retried;
unresolved judge errors remain missing, never answer-quality zeros. Judge
requests may run concurrently; generation stays serial. Account prices are
unknown; token usage is recorded without substituting a zero invoice amount.

Codex will source-check the same prespecified five pilot question IDs across
four variants (20 answers), all execution/judge errors, and flag questionable
extra assertions. Those source adjudications remain distinct from original
DeepSeek judgments. No human scoring or significance claim is made. Publish
execution denominators, missingness, quality, exploratory latency, physical
attempt usage, costs and failure examples, including inconvenient results.

## Commands

```powershell
.venv/Scripts/python.exe -m app.evaluation.cli --provider dashscope --embedding-provider dashscope --chat-model qwen3.7-flash-2026-07-15 --real-manifest benchmarks/real-v1/corpus-manifest.json --split dev --allow-ai-reviewed --limit 20 --order-seed 20261018 --structured-output json_schema --max-output-tokens 4096 --capture-responses --ledger artifacts/evaluation/real-v1-dev-full/ledger.sqlite --output-dir artifacts/evaluation/real-v1-dev-full/generation
.venv/Scripts/python.exe scripts/judge_development.py --run-dir artifacts/evaluation/real-v1-dev-full/generation --ledger artifacts/evaluation/real-v1-dev-full/ledger.sqlite --output-dir artifacts/evaluation/real-v1-dev-full/judgments --execute
```

If generation is interrupted, use the identical command with `--resume`. An
existing completed or failed executed unit is not silently regenerated. Any
changed generation settings require a new directory. Any new experiment after
seeing this development run must be identified separately before test freezing.
