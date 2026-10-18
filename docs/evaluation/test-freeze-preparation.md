# Held-out test preparation after the final development regression

This is the historical preparation record: at recording time, **the experiment
was not frozen and no held-out answer had been generated or judged**.
The subsequent [freeze record](../../benchmarks/real-v1/frozen-experiment.json)
now binds the implementation and design; see the [runner documentation](frozen-experiment.md)
for current commands. Held-out generation and judging have not started.
[test-preparation.json](../../benchmarks/real-v1/test-preparation.json)
records the selected generation configuration, proposed analysis and review
sample before any test outputs. The paid development CLI still rejects test
execution. This preparation record does not unlock it; the separate committed
freeze is required. Its verification command checks this historical preparation
status rather than the current experiment's execution state.

## Choices recorded now

Keep the final-rule v8 generation behavior from clean commit `c293dc0`: fixed
Qwen snapshot, 4096 output cap, shared exact citation checks, request-specific
evidence ID enums, bounded researcher/answer corrections and existing retrieval
limits. Full development results, hashes and costs remain in their separate
packet. Do not continue tuning for a better development score before this main
comparison; a necessary implementation repair must be identified explicitly.

The main matrix remains **40 test questions × four workflows × three fresh
repeats = 480 units**. Generation stays serial; repeat order seeds are
`20261019`, `20261020`, `20261021`. Each repeat uses a new answer generation,
with shared index setup reported separately. Optional cross-model generation
is excluded from this execution stage and needs a separate protocol before its
answers are generated. DeepSeek remains the delegated blinded scoring service.

The primary comparison is multi-agent versus single-agent RAG: mean normalized
key-point correctness and additional answer-request cost. Executed failures
receive quality zero in the primary test comparison, with conditional successful
quality reported separately. Unexecuted units and judge failures remain missing,
not zeros. The paired primary analysis requires all three known repeat outcomes
for both workflows; disclose exclusions and missing-value sensitivity bounds.
Unknown charges stay null. Keep setup, per-answer generation and judging costs
separate. Other workflow comparisons are secondary.

Average repeats within each question before paired differences. Resample the
eight test document families, using 10,000 bootstrap replicates, seed
`20261023`, and two-sided percentile 95% intervals. With only eight families,
interpret intervals as exploratory and show family-level differences; 480
trace rows are not 480 independent questions. Freeze the implementation of
these calculations before test execution, not after observing the outcomes.

The judge uses the same exhaustive key-point and answer-block semantics
calibrated on development, with blinded workflow/model/cost/timing labels and
at most one format correction. Do not apply the development-only reference
overlay to test questions. Preserve raw judgments and explicit source
adjudications. AI review remains separate from human review; there is no
established human agreement rate or claim of independent human validation.

## Prespecified source review

Select repeat index **0**, four workflows for each of **20 question groups**,
for 80 source reviews. The fixed sample seed is `20261022`. Draw two questions
from each of the ten task-type/language strata, accepting the first draw that
covers all eight test families. This selected the first draw. Selection used
only IDs and group/family/type/language metadata, never answers or expected
winners. The exact IDs and strata are saved in the JSON record. Additional
failure or disagreement reviews must remain separate from this sample.

This is the delegated AI review sample. Human review is still pending and is
never imputed from Codex or DeepSeek checks.

## Work required for the final freeze

Implement and validate the frozen test runner and test judge, then bind their
hashes together with code, prompts, source/data/review identities, effective
configuration, analysis rules and the review sample. Repeat identity must be
explicit. A resumed process must not repeat completed paid units, and must
attribute any repeated setup separately from query requests. The current
development publisher's timestamp boundary assumes one uninterrupted setup;
it is insufficient as the only phase marker for a resumed multi-repeat run.

Prove these orchestration and accounting contracts offline, including gold
isolation, before the first held-out request. Commit the final freeze record
before execution. The generation candidate above does not substitute for that
completed freeze.

Reproduce the sample and validate the preparation identities without API calls:

```powershell
.venv/Scripts/python.exe scripts/verify_test_preparation.py
```
