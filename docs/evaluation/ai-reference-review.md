# Delegated AI reference review

On 2026-10-18 the user delegated reference review to Codex and/or DeepSeek.
This review is recorded as AI review throughout. It does not create human
approval records or claim human-calibrated model scores.

The initial DeepSeek pass used 16 accounted official CLI tasks: 12 complete
document-family reviews covering all 48 answerable questions, and four
whole-pool shards covering all 12 insufficient-evidence questions. Every
document in the selected split was supplied in full to its absence shard;
no document was truncated. Model responses were checked for coverage,
duplicate IDs, source membership and literal suggested quotes. Five task
outputs contained at least one non-verbatim suggested quote, which was
rejected for automatic application and retained with an explicit flag.

The reviewer requested changes to 22 answerable references. These are
advisory findings, not 22 established factual errors: several concern
incomplete selected gold passages or overly strict interpretations of a
valid paraphrase. Codex adjudicates them against the pinned source files.
The initial findings and exact request/response hashes are preserved in
[`initial-audit.json`](../../benchmarks/real-v1/review/initial-audit.json).
The final source adjudication and every corrected reference are preserved in
[`final-audit.json`](../../benchmarks/real-v1/review/final-audit.json) and
[`corrections.json`](../../benchmarks/real-v1/review/corrections.json).

The helper retains its default 32,000-byte prompt bound. This full-source
audit explicitly opts into a bounded 131,072-byte limit per task, while
preserving a single physical model request and 2,048 output-token cap.
Usage remains in the durable local ledger; account-mode financial totals
are unknown. The author and reviewer share the DeepSeek Flash alias in
fresh contexts; Codex provides a second model's adjudication. This is not
independent human validation and may retain correlated model errors.

The completed AI adjudication accepts **60/60 references for the dev pilot**.
Of these, 46 cases changed: 44 have improved evidence, 20 have corrected or
better scoped key points, and nine have revised question wording (categories
overlap). One test question was replaced because its integer-coercion fact
was too similar to a dev question. All 139 final gold evidence units match
their literal locations and are reachable after actual document normalization
and default chunking (1,200 characters, 200-character overlap). Whole-pool
absence checks cover every split document; none establishes the requested
deployment-specific measurement for the 12 negative questions.

Some DeepSeek concerns were rejected after source checking, including a
claimed absence of explicit strict/lax ranking despite the quoted ordered
list, a non-existent three-gold-quote limit, and a claim that a selected v1
passage omitted `always=True` when it contained that text. No reviewer quote
was copied into gold without literal source validation. Feedback remains in
the audit trail even where Codex disagreed. DeepSeek's stored reference hash
identifies the version it saw; the final Codex decision binds the final
reference hash, including any subsequent source-checked corrections.

The review used 42 physical CLI requests across initial and follow-up passes:
583,225 input tokens and 23,244 output tokens, with 12,544 cached input tokens
included in the input count. Resumed responses did not incur new requests.
The monetary amount remains unknown for account mode. These are reference
curation costs, not workflow benchmark usage or latency measurements.

The user-authorized path forward is an **AI-reviewed dev pilot** using
`--allow-ai-reviewed`. It requires a complete, current AI audit and labels
AI and human review separately in run metadata. Human approval remains 0/60;
it is an optional later independent check. Paid held-out test execution still
requires the frozen experiment runner. No workflow-quality score or human
calibration result is produced by reference review.

Validation: **584 offline tests passed; one live test deselected**. The
repository verifier passed, the original 120 fake traces passed their
quality gate, and a 20-trace AI-reviewed dev smoke run correctly recorded
AI review as true and human approval as false. The regenerated review page
rendered 60 cards in headless Edge with the completed-AI-review label.
