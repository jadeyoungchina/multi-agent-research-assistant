# Delegated AI reference review

On 2026-10-18 the user delegated reference review to Codex and/or DeepSeek.
This review is recorded as AI review throughout. It does not create human
approval records or claim human-calibrated model scores.

The initial DeepSeek pass used 16 accounted official CLI tasks: 12 complete
document-family reviews covering all 48 answerable questions, and four
whole-pool shards covering all 12 insufficient-evidence questions. Every
document in the selected split was supplied in full to its absence shard;
no document was truncated. Model responses were checked for coverage,
duplicate IDs, source membership and literal suggested quotes. Six task
outputs contained at least one non-verbatim suggested quote, which was
rejected for automatic application and retained with an explicit flag.

The reviewer requested changes to 22 answerable references. These are
advisory findings, not 22 established factual errors: several concern
incomplete selected gold passages or overly strict interpretations of a
valid paraphrase. Codex adjudicates them against the pinned source files.
The initial findings and exact request/response hashes are preserved in
[`initial-audit.json`](../../benchmarks/real-v1/review/initial-audit.json).
Corrections and the final review will be committed separately.

The helper retains its default 32,000-byte prompt bound. This full-source
audit explicitly opts into a bounded 131,072-byte limit per task, while
preserving a single physical model request and 2,048 output-token cap.
Usage remains in the durable local ledger; account-mode financial totals
are unknown. The author and reviewer share the DeepSeek Flash alias in
fresh contexts; Codex provides a second model's adjudication. This is not
independent human validation and may retain correlated model errors.
