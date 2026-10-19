# Long-running execution and publication

The experiment runs serially and can outlast an interactive chat turn. The local
supervisor continues the **already authorized** frozen generation, then blinded
judging and the initial paired analysis. It publishes a generation packet after
each 160-unit repeat, pushes the existing feature branch and verifies both push
and PR CI before starting more paid work. It never merges the default branch.

The helper scripts in [tools](tools/) were added after the scientific freeze.
They manage process lifetime and publication only; frozen generation, rubric,
reference keys, sample and primary statistics remain in the hash-bound original
modules. They do not introduce alternate experiment parameters.

Read-only preflight and local status:

```powershell
.venv/Scripts/python.exe docs/evaluation/heldout-test/tools/continue_run.py
.venv/Scripts/python.exe docs/evaluation/heldout-test/tools/continue_run.py --status
```

Execution requires the explicit `--execute` flag. An OS lock prevents two
supervisors. An existing attached generation process is allowed to finish
without interruption. Completed units resume from saved records; an uncertain
paid unit, modified working tree, wrong branch, failed push or failed/unconfirmed
CI stops the supervisor. No uncertain paid unit is automatically rerun.

Local state and phase logs are under
`artifacts/evaluation/frozen-run-control/`; raw records remain under
`artifacts/evaluation/real-v1-test/`. These runtime directories are ignored by
Git. Public packets contain only allowlisted answers, evidence, metadata and
request receipts; no credentials, HTTP headers or local account files.

The supervisor finishes at **initial analysis, source review pending**. The
prespecified 80-answer AI source review and any additional error reviews still
need explicit trace-bound adjudications. Original judgments must remain intact.
Preliminary analysis is labeled as such; human scoring remains zero. A later
chat should inspect saved state and continue this review rather than regenerate
completed answers or relabel the preliminary output as final.
