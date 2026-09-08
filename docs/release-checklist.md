# Release checklist

Use this checklist for each candidate revision. Unchecked items are actions to
verify, not an assertion that a remote release or real-provider run exists.

## Local candidate

- [ ] Review `git status --short` and staged changes; exclude private documents, keys, databases, and output artifacts.
- [ ] Keep `DASHSCOPE_API_KEY` and `OPENAI_API_KEY` blank in `.env.example`.
- [ ] Run `python -m pytest -m "not live" -q` using Python 3.11 and Node.js 22.
- [ ] Stage intended source/docs, then run `python scripts/verify_repository.py` so new files are included in the tracked-file check.
- [ ] Run `python -m app.evaluation.cli --provider fake --variants all --output-dir artifacts/evaluation/fake --enforce-gate`.
- [ ] Inspect the generated 120-trace matrix, failures, safe configuration, digests, and gate result; retain outputs locally or as CI artifacts.
- [ ] Run `git diff --check` and `git diff --cached --check`.
- [ ] Start fake mode; upload demo documents, ask a question, inspect citations and terminal progress.
- [ ] Confirm bilingual reconstruction disclosures, independent-project disclaimer, non-commercial restriction, and exact license copies.
- [ ] Check current attribution paths against adapted-file headers; do not restore nonexistent planned filenames.

The verifier reads working-tree content for paths returned by `git ls-files -z`.
It ignores untracked local data and does not inspect old Git objects or guarantee
that the index and working tree are identical. Review the staged diff before
commit; runtime placeholder directories allowed by `.gitignore` must not be
tracked. Automated detection covers common provider-key formats, not every
possible private value or document disguised as source code.

## Real-provider evidence

- [ ] If required for publication, run and record a DashScope `qwen3.7-flash` smoke check with local credentials and a non-sensitive corpus.
- [ ] For any numerical real-model claim, preserve generated run reports, exact configuration, source/dataset digests, prompt version, and repeated trials.
- [ ] Review outputs for private source text before sharing; never commit evaluation output or credentials.

Offline CI intentionally cannot certify model availability, regional endpoint
access, live answer quality, or billing behavior. Missing real-provider evidence
must be reported as unverified, not replaced by fake results.

## History and publication

- [ ] Preserve the authorized pre-rewrite Git bundle and verify it before changing history.
- [ ] Preserve exact disclosures while reconstructing milestone commits; respect upstream source availability dates.
- [ ] Verify the final commit tree and run offline checks against it.
- [ ] Confirm the expected remote commit before an authorized `--force-with-lease` push.
- [ ] Create/push the intended `v1.0.0` tag only after final release checks.
- [ ] Check remote CI and its uploaded fake-benchmark artifact, and record publication status separately.

CI installs the two requirements layers, runs offline pytest, the repository
verifier, and the fake gate, then attempts artifact upload even if an earlier
step failed. This checklist does not itself authorize or perform publication.
