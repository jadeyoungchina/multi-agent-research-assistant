# Frozen held-out experiment runner

The runner, held-out judge and primary analysis are implemented. The calibration
record and committed freeze are required before paid held-out execution. The
development CLI continues to reject paid test runs.

## Identity and execution

The freeze binds code/data/test design hashes, effective provider configuration,
Python major/minor and pinned package versions, the local DeepSeek CLI hash,
review sample, generation order and primary statistics. Source code is hashed
with LF-normalized line endings; byte-addressed corpus/data snapshots retain
exact bytes. Unrelated documentation can be committed without changing frozen
inputs. Changes to protected implementation require the original frozen checkout
or a separate, explicitly identified experiment.

The 40 test questions use all 16 test documents as the same candidate pool.
Only case ID, question and that complete source list reach a workflow. Evaluator
reference keys never enter generation. Existing v8 generation logic remains
unchanged; the workflow builder adds a separate index client/ledger only for
this runner, and the shared block-rubric extraction preserves the development
prompt byte for byte.

Each repeat has a distinct identity, order seed, answer ledger, index ledger,
started/completed journal and setup-session history. Generation takes one OS
lock across repeats. Completed repeats resume without constructing providers.
The final report can be rebuilt from fully journaled answers with no paid calls.
Unknown charges remain unknown. Index rebuilds are retained separately from
answer requests, including failed setup sessions.

A started unit interrupted before durable completion stops automatic generation.
`--resolve-interrupted-as-failed` retains it as a failed execution with every
recorded receipt; it never reruns the uncertain unit. Its unknown latency/retry
placeholders are excluded from performance aggregates, with valid denominators
and authoritative nulls saved in `performance.json`. Torn final trace bytes are
quarantined before repair; a torn completion still requires interruption
resolution. Returned model drift is retained and halts subsequent generation.

## Judging and analysis

The test judge uses original test reference keys, the exhaustive block rules,
8192 output tokens and at most one format correction. It does not load the dev
reference overlay. Cached judgments are validated before new paid jobs. Each
repeat has its own judge ledger. Judging a later subset preserves earlier
judgments and summary counts. The CLI alias `deepseek-flash` has no verified
immutable model snapshot; fixing the client hash does not eliminate provider
model drift. Human calibration/scoring is still zero.

Before freeze, exercise the test schema on the 20 already source-reviewed dev
answers. The prespecified structural gate is at least **16/20 valid judgments**;
all errors and raw attempts remain visible. Also report exact key-point and
block-verdict agreement against the existing **AI** source labels, with valid
and planned denominators. This gate checks operability, not human agreement or
proof of semantic reliability. No test question is generated for calibration.

The primary analysis requires all three generation repeats. It scores executed
failures as zero, while missing/error judgments remain missing. It averages
repeats within each question before paired comparisons and reports whole-matrix
missing-value bounds. Quality and additional answer cost use family bootstrap
intervals with the recorded seed. Empty bootstrap draws are disclosed; fewer
than two informative families yields no interval. Unknown invoice amounts and
missing ledgers cannot become zero costs. Index and judging costs stay separate.

AI source adjudications live in a separate `adjudications.json` under the run
directory, bound to the freeze ID and each exact trace hash. Accept/correct
decisions preserve original outputs; failed executions use separate failure
audits. The prespecified 80-answer sample includes those failure audits without
mixing extra disagreement reviews into sample completion.

## Commands

After committing implementation and publishing development-only calibration:

```powershell
.venv/Scripts/python.exe scripts/frozen_experiment.py create
```

Commit the generated `benchmarks/real-v1/frozen-experiment.json` before execution.
The following validate without making a paid request:

```powershell
.venv/Scripts/python.exe scripts/frozen_experiment.py verify
.venv/Scripts/python.exe scripts/frozen_experiment.py generate
```

Execute serial generation, then blinded judging and primary analysis:

```powershell
.venv/Scripts/python.exe scripts/frozen_experiment.py generate --execute
.venv/Scripts/python.exe scripts/frozen_experiment.py judge --execute
.venv/Scripts/python.exe scripts/frozen_experiment.py analyze
```

Use `--repeat 0`, `1` or `2` to work on an individual repeat. Resume uses the same
command and identity. Preparation/calibration costs have their own ledger and
are not silently pooled with test answer or judge costs.
