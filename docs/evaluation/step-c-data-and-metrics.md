# Stage C: public-document data and metric corrections

Date: 2026-10-18. This milestone provides 24 pinned public documents, 12
topic families, and 60 AI-authored candidate questions (20 dev / 40 test;
30 Chinese / 30 English). Human reference approval remains **0/60**.
It does not publish model-quality or admissions-outcome claims.

Subsequent user-delegated AI review is complete: 60/60 references accepted
for a dev pilot, with 46 cases improved. The detailed source adjudications,
before/after records and independent-context DeepSeek feedback are in
[`ai-reference-review.md`](ai-reference-review.md). Human review remains
separate; the authorized pilot uses `--allow-ai-reviewed` with a current
audit, rather than waiting for a human approval record.

The data card and immutable source records are in
[`benchmarks/real-v1`](../../benchmarks/real-v1/README.md). Every reference
quote is checked against its recorded line range. Candidate pools belong
to split manifests, not gold annotations. Limiting the question count leaves
the complete split corpus available, including distractors.

Metric version `global-first-pass-v2` takes the actual first-round global
top five before checking source membership. Each retrieval round records
the queries actually embedded and its fused ranking. Final-context coverage
and recall over all rounds have distinct fields. Baseline retrieval/citation
metrics and empty-gold retrieval recall are N/A, with applicable denominators.
`citation_id_validity` is explicit; `citation_precision` remains a deprecated
compatibility alias and does not measure semantic support. Lexical answer
scores also remain diagnostics. `app/evaluation/semantic.py` supplies the
blinded semantic rubric, per-key-point judgments, claim-support and citation
completeness denominators, abstention rules, and validation of judge output.
ID-valid but irrelevant citations receive no support credit. Actual model
judging and human calibration remain separate operations and have not run.

Critic follow-up queries now precede old plan expansions so the configured
query cap does not silently discard them. Critic criterion coverage is judged
semantically, with an explicit instruction to identify unmet criteria;
deterministic checks still reject missing findings and invalid supporting
IDs. The former substring test incorrectly rejected faithful paraphrases.
These changes precede any paid held-out evaluation.

Run counters now count physical attempts and retain failed provider usage in
the saved run while preserving the previous business-state checkpoint.
Evaluation traces preserve attempt IDs and usage completeness. Multi-agent
answer scoring excludes the renderer's copied source bibliography, which
could otherwise inflate answer coverage. All RAG prompts request explicit
statement-level citation tokens.

Validation: the source/hash/license/split checks pass, 80 dev fake traces
completed, and the original 120 fake traces passed their regression gate.
Regression tests cover global rank errors, first-versus-later rounds, N/A
denominators, whole-pool selection after limiting, source tampering, split
leakage, pending human review, query truncation, paraphrases, and failed
request accounting. CI also exercises a five-question real-document fake
run. Fake output proves pipeline execution only.

DeepSeek drafted the initial candidates through 12 accounted official CLI
requests. Account-mode prices were not returned by the official catalogue,
so monetary totals remain unknown. The token ledger stays local; sanitized
task hashes and request IDs are in `drafting-provenance.json`. AI source
checks and curation are recorded as AI work, not human review.

The human review packet can be generated with
`python scripts/review_real_dataset.py`. It contains all 60 cases, exact
source locations, surrounding source text, and whole candidate-pool links.
The page was rendered in a separate headless Edge profile: 60 question cards
rendered and the screenshot was visually checked. Explicit exported reviews
are tied to dataset/manifest hashes; attribution and timestamps are required.
Generating or opening the page approves nothing. Further tests reject stale,
duplicate, unattributed and incomplete review imports before modifying data.

Stage D remains outstanding: five-case real-model
dev calibration, judge calibration, full dev runs and prompt/parameter freeze,
then the prespecified held-out repeated comparison and statistical analysis.
Human reference checks are an optional later independent validation under
the user's delegated-AI-review choice.
The initial drafting batch yielded exactly 60 candidates rather than the
design's approximate 80; rejected or deficient cases can be replaced during
reference curation, before any paid held-out evaluation.

Final local verification for this milestone: **566 offline tests passed, one
live test deselected**. The repository verifier passed after all source
snapshots were staged. The first data/metric commit `cf25b36` also passed
GitHub Actions, including the added real-document pipeline check.
