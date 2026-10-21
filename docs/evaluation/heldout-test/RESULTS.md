# Frozen held-out results with AI source review

**480/480 executions recorded: 473 successful, 7 failed.** Prespecified source review: **80/80**, plus 17 additional error/failure reviews. Human review/scoring: **0**.

These are exploratory, AI-assisted results on 40 documentation questions across eight document families, with three fresh repeats. Source checks use the same DeepSeek alias in separate contexts, with explicit Codex follow-ups; they are not independent human validation.

Independent human review is excluded from the current delivery at the user's request (2026-10-20), rather than a pending acceptance step. Human-review counts remain zero. Frozen metadata and historical method notes retain their original status fields as provenance.

[Frozen protocol](../frozen-experiment.md) · [Initial analysis before source review](INITIAL-ANALYSIS.md) · [Audit method](SOURCE-REVIEW-METHOD.md) · [Recomputed analysis](reviewed-analysis.json) · [Summary](reviewed-summary.json) · [Trace-bound decisions](source-adjudications.json) · [Raw source reviews and literal proofs](source-review-records.json) · [All safe receipts](all-request-usage.jsonl)

The [executed analysis notebook](../../../output/jupyter-notebook/real-evaluation-analysis.ipynb)
recomputes tables and [three figure sets](figures/analysis-tables.json) from the public packet.
See the [offline reproduction guide](REPRODUCE.md) and [seven selected failure cases](FAILURE-CASES.md).

| Workflow | Successful / 120 | Known three-repeat questions / 40 | Primary quality | Conditional citation support | Conditional citation completeness | Median successful seconds | Answer attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_llm | 120 | 40 | 0.643 | N/A | N/A | 5.961 | 120 |
| llm_rag | 119 | 40 | 0.919 | 0.856 | 0.851 | 3.973 | 248 |
| single_agent_rag | 119 | 40 | 0.821 | 0.757 | 0.740 | 8.646 | 815 |
| multi_agent_rag | 115 | 40 | 0.878 | 0.561 | 0.519 | 49.848 | 2070 |

Plain RAG has the highest observed primary quality in this small question set (0.919); multi-agent RAG is 5.8 times slower than single-agent RAG by median successful latency. The prespecified primary comparison is multi-agent versus single-agent, not a claim that multi-agent beats every baseline.

![Quality, latency and physical answer attempts](figures/workflow-comparison.png)

Primary quality averages the three repeats within a question, including executed-failure zeros. A question with a missing successful-answer judgment is excluded and disclosed. Citation metrics condition on applicable, successfully scored answers; baseline citations are N/A. Exact denominators and missingness are in the summary. Structure validity is separate from entailment.

The prespecified multi-agent minus single-agent difference is **0.057**, with 40/40 complete question pairs and exploratory family-bootstrap 95% interval **[0.013888888888888895, 0.09722222222222222]**. Whole-matrix missing-value bounds: [0.05694444444444444, 0.05694444444444444].

The primary comparison, sample, seeds and eight-family bootstrap were fixed before generation. The initial result and every original judge error remain available. An interval containing zero does not establish a reliable improvement; a positive estimate on this dataset alone does not establish general superiority.

Source decisions: 39 accepted, 51 corrected; 37 explicit Codex follow-ups. Successful answers still missing judgments: 0. Original review errors and any bounded format repairs remain in the raw records.

All phases retain **3960 physical requests**. Index construction, answer generation, original judging and source review are separate in the summary. Unknown usage and unverified account prices remain null. Shared index setup and judge/review costs are excluded from the paired incremental answer-cost estimate; dated API estimates are not invoices.

Generation packets retain exact answers/evidence: [repeat 0](repeat-0/summary.json), [repeat 1](repeat-1/summary.json), [repeat 2](repeat-2/summary.json). Failed executions are preserved with safe diagnostics. This completes the delegated AI experiment stage. Broader benchmarks are outside this study; subsequent human review is excluded from the current delivery.

Recompute identities, original-judge preservation, all scores, request costs and the frozen paired statistic without paid calls:

```powershell
.venv/Scripts/python.exe docs/evaluation/heldout-test/tools/reproduce_frozen.py
```
