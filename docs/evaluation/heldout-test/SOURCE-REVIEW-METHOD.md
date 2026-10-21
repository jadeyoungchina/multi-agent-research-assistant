# AI source review after initial judging

The initial 480-unit generation and judging are complete and retained separately.
The frozen primary calculation has 34/40 complete question pairs, 15 missing
successful-answer judgments and seven execution failures. The prespecified
80-answer review sample has not yet been checked. Review makes no changes to
generation, reference keys, the frozen rubric, primary statistics or failures.

The review targets exactly the **80 answers selected before generation**, plus
all original judge errors and all execution failures outside that sample.
These yield **97 unique targets: 80 sampled and 17 additional**. The overlap is
counted once. Two of the 80 sampled executions failed; their records receive
failure audits rather than invented answer scores.

The user authorized AI review and official DeepSeek CLI delegation. The helper
uses the same `deepseek-flash` account alias in **separate contexts**, with no
original judge labels, workflow, generation model, cost or latency supplied.
This is a second AI check, **not independent human validation or an independent
model replication**. Human review remains zero, and alias drift/correlated model
bias remain limitations. This proof-audit prompt was written after seeing the
initial aggregate analysis; it is not described as a preregistered prompt.

Each successful-answer audit applies the unchanged `test-block-v2` rubric and
must supply literal pinned-source quotes for every answerable reference point.
Each factual answer block rated supported/partial additionally needs a literal
quote from a passage actually cited by that block. Programmatic checks reject
invented source quotes, missing reference checks, borrowed citation IDs and
schema/citation inconsistencies. Exact substring membership does **not** prove
entailment; the AI still makes a semantic judgment. Failure audits inspect only
retained error codes and allowlisted response diagnostics, leaving uncertain
causes explicitly uncertain and adding no semantic score.

The helper binds its code/CLI hashes, frozen identity, exact trace hashes, original
judgment hashes and the immutable initial analysis before paid calls. There are
three workers, an 8192-token output cap and at most one explicit format/proof
correction. Transport errors are not silently retried. Saved tasks and raw
attempts are retained; resume cannot silently repeat incomplete paid work.
Review has a separate ledger and its account price remains unknown.

When source labels match the original labels, an accepted adjudication preserves
the entire original judgment, including its reasons. Changed labels or recovered
judge errors receive explicit corrected adjudications. Any subsequent Codex
source decisions must have an exact trace hash and concrete reason and must
remain separately attributed. The original judging files and published initial
analysis are never overwritten.

Read-only preflight, execution and materialization:

```powershell
.venv/Scripts/python.exe docs/evaluation/heldout-test/tools/check_source_review.py
.venv/Scripts/python.exe docs/evaluation/heldout-test/tools/source_review.py
.venv/Scripts/python.exe docs/evaluation/heldout-test/tools/source_review.py --execute
.venv/Scripts/python.exe docs/evaluation/heldout-test/tools/source_review.py --materialize
.venv/Scripts/python.exe scripts/frozen_experiment.py analyze
```

The reviewer publishes both the original and source-reviewed paired analyses,
denominators, all retained review errors, exact evidence and separate request
receipts. Any unresolved source checks remain incomplete rather than counted as
completed. An exploratory interval over only eight document families and AI
references cannot establish general architectural superiority.
