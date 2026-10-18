# Final-rule 20-question development regression

**80/80 successful executions**; **80/80 valid AI scores**. Codex source checks: 21/80; human scoring: **0**.

This is a separate final-rule development regression after tuning on the same questions. It is not pooled with earlier runs or treated as a held-out comparison. The 40 held-out questions have not been generated or judged. No significance or general architecture-superiority claim follows from these results.

[Prespecified protocol](../final-dev-protocol.md) · [Summary](summary.json) · [Answers and exact evidence](traces.jsonl) · [Original AI judgments and adjudications](ai-judgments.json) · [Physical receipts](request-usage.jsonl)

## Generation and semantic quality

| Workflow | Executions | AI scores | Key-point correctness | Full key points | Citation block support | Citation block completeness | Median successful seconds | Chat attempts | Query embeddings | API estimate CNY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_llm | 20/20 | 20 | 0.738 | 0.692 | N/A | N/A | 5.811 | 20 | 0 | 0.0072358 |
| llm_rag | 20/20 | 20 | 0.892 | 0.808 | 0.801 | 0.752 | 5.087 | 20 | 20 | 0.0117296 |
| single_agent_rag | 20/20 | 20 | 0.854 | 0.808 | 0.873 | 0.858 | 9.964 | 42 | 93 | unknown |
| multi_agent_rag | 20/20 | 20 | 0.875 | 0.800 | 0.786 | 0.713 | 40.953 | 122 | 194 | 0.1007926 |

Correctness is the per-answer mean of required key-point scores (0/1/2), normalized to 0–1. Full key points counts points receiving 2. Metrics are macro means over valid, applicable judgments; all denominators and missing judgments are explicit in summary.json. Execution success means the pipeline returned under shared strict inline citation checks. Structural validity does not establish semantic correctness or citation entailment.

Citation metrics audit every deterministic Markdown block, including uncited prose and code. Full support requires all domain assertions in a cited block to follow from its attached exact passages. Scoped missing-evidence limitations are N/A. Baseline citation metrics are N/A. These block denominators differ from the five-question pilot's model-selected atomic claims; do not compare their citation rates directly. ID validity alone is not entailment.

## Abstention and retrieval diagnostics

| Workflow | Correct abstention | False abstention | Answers with unsupported assertions | First-round recall@5 |
| --- | ---: | ---: | ---: | ---: |
| baseline_llm | 0.000 (n=4) | 0.000 (n=16) | 0.600 (n=20) | N/A (n=0) |
| llm_rag | 0.750 (n=4) | 0.000 (n=16) | 0.050 (n=20) | 0.769 (n=16) |
| single_agent_rag | 1.000 (n=4) | 0.062 (n=16) | 0.050 (n=20) | 0.665 (n=16) |
| multi_agent_rag | 0.750 (n=4) | 0.000 (n=16) | 0.050 (n=20) | 0.769 (n=16) |

Retrieval uses the original global first-round top five and original gold spans. A reference can prescribe one of several valid implementations; recall is a diagnostic rather than semantic answer correctness. Correct abstention means declining an unanswerable measurement without an invented conclusion; false abstention applies only to answerable questions.

## Supplementary citation-format audit

This shared structural check was fixed before the final-rule regression. Its results are separate from semantic support and do not relabel earlier executions. All RAG answers are checked with the same exact token syntax, retrieved-ID and declared-inline consistency rules; failed executions and closed-book citations are N/A.

| Workflow | Exact syntax | Syntax and reference consistency | Applicable successful answers |
| --- | ---: | ---: | ---: |
| baseline_llm | N/A | N/A | 0 |
| llm_rag | 1.000 | 1.000 | 20 |
| single_agent_rag | 1.000 | 1.000 | 20 |
| multi_agent_rag | 1.000 | 1.000 | 20 |

## Cost, failures and provenance

Generation commit: `c293dc02979af4b95a9d37f458480eaf66f3c56c`; prompt `evaluation-v8`. All chat roles use `qwen3.7-flash-2026-07-15`, strict field output, temperature 0, disabled thinking, cap 4096, two provider retries, at most two critic loops and one researcher and one answer/writer citation correction. Real text-embedding-v4 vectors have 1024 dimensions, chunks 1200/200 and top-k 5. All RAG variants search the same complete eight-document dev pool. Generation is serial with the prespecified shuffled/rotated order and seed 20261018.

Shared index setup: 14 embedding attempts; 0.014885 CNY estimated; 56159 ms. Setup is excluded from answer latency and variant costs. Variant costs include all recorded answer attempts, retries and repairs. Failed requests with unknown usage remain unknown; no zero charge is invented. Dated API price estimates are not invoices.

The combined ledger records 611 physical requests. Judge requests: 86; separate preparation requests: 0. DeepSeek official CLI account-mode pricing is unverified, so judge and combined monetary totals remain unknown. Raw responses and any single bounded format repair remain separate. Model aliases can drift for the judge; generation uses the fixed snapshot above.

Execution failures: 0; unresolved judge errors: 0; Codex-corrected judgments: 12. Failures remain missing for semantic averages, with their execution denominator visible. The original judge output is never overwritten by a source adjudication.

[Failure diagnostics](failure-diagnostics.json) preserve only allowlisted generated response captures associated with failed traces. They contain public-corpus model text and validation locations, not provider HTTP headers or credentials.

The two scoring-only reference clarifications, source review and local runtime check were recorded before full generation. The original dataset and historical pilot references remain unchanged. Codex source checks cover successful answers for the prespecified five pilot IDs across all workflows; execution failures are audited separately. This is AI review, not human validation. Remaining judge errors and correlated AI biases may remain even with schema-complete output. Public documentation may have appeared in model pretraining. Latency is exploratory on one shared machine.

Recompute published hashes, scores, denominators and request costs without API calls:

```powershell
.venv/Scripts/python.exe scripts/verify_full_dev.py
```

This records the complete development run, including its unresolved limitations. The final-rule development regression is complete. Experiment freeze, repeated held-out testing, independent human validation and any final comparison remain separate future work.
