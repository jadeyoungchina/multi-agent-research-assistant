# Default startup repair and historical experiment reproduction

Product maintenance, 2026-10-21. This change repairs application composition after
the published experiment and keeps the original study separately reproducible.
It does not produce a new real-model result.

## Startup failure and regression

The documented `app.main:app` entry calls `build_container()` without an explicit
settings object. The builder loads configuration into `active_settings`, but the
Researcher and Writer repair limits incorrectly read the nullable `settings`
argument. Valid loaded configuration therefore still fails with:

```text
AttributeError: 'NoneType' object has no attribute 'max_researcher_citation_repairs'
```

The new [default-entry integration regression](../../tests/integration/test_product_smoke.py)
uses the same no-container startup path with a valid, isolated fake configuration.
It failed before the fix. Reading both limits from `active_settings` makes startup,
health and upload succeed. Explicitly supplied configurations retain their original
behavior; the frozen experiment supplied its settings explicitly.

```bash
python -m pytest tests/integration/test_product_smoke.py::test_default_entrypoint_uses_loaded_settings -q
```

The [separate process acceptance helper](tools/check_default_startup.py) starts the
documented Uvicorn target, uses real loopback HTTP requests for upload/research/SSE,
and checks the report and citations. It supplies a fake-only profile with isolated
storage and empty model credentials, bypasses proxy environment variables for its
loopback HTTP client, preserves the UI content security policy, and stops its child
server after the check. [Recorded acceptance evidence](default-startup-check.json)
contains the product commit and code hashes, without local paths or credentials.

```bash
python docs/engineering/tools/check_default_startup.py
```

## Reproduce the study with its original implementation

The historical experiment keeps freeze identity
`0fc19f9d47679eaf55ebeddd7c6857b36779b632db6d5dd644cfda6c6aeb3270`.
Its complete source and analysis snapshot is Git commit
`7e0cf6cdf167e112775b1376458b1e7ceff1af1a`. That snapshot contains the exact 257
frozen inputs, source audits, published packets and executed analysis tools.

```bash
python docs/evaluation/heldout-test/tools/reproduce_frozen.py
python docs/evaluation/heldout-test/tools/reproduce_frozen.py --notebook
```

The launcher first compares current public result bytes against the original Git
snapshot, so an edited answer cannot be hidden by checking an older copy. It then
creates or reuses a detached worktree at the pinned commit and runs the original
verifier or notebook. It rejects a different commit, unexpected tracked/untracked
edits, changed analysis helpers and changed notebook cell sources. Regenerated
outputs can vary by platform; executable sources remain pinned. Existing edits are
left in place on rejection.

The [regressions](../../tests/evaluation/test_frozen_reproduction.py) cover modified
published answers, edited verifiers and edited notebook sources. The notebook
source guard was also observed failing before implementation. The original
frozen-run validation still refuses changed code: it is not weakened or re-frozen.
No paid generation is part of reproduction, and original successful answers,
failures, judgments, prices and unknown values remain the study's evidence.

The current notebook also selects this original snapshot before importing its
evaluator; it can still execute all seven cells from a fresh kernel. See the
[reproduction guide](../evaluation/heldout-test/REPRODUCE.md) for dependencies and
output locations. Subsequent human review remains outside the delivery scope.
