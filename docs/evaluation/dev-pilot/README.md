# Five-question live development pilot

Final matrix: **20/20 successful executions**, **20/20 valid AI scores**. Codex source checks: 20/20; 15 judge outputs corrected with explicit reasons. Human scoring: **0**.

This is development calibration on the same five prespecified questions, not a held-out estimate or evidence that one architecture wins. Earlier failures are retained separately. The 40 held-out questions have not been run.

[Selection and tuning decisions](../dev-pilot-selection.md) · [Machine-readable summary](summary.json) · [Final answers and evidence](final-traces.jsonl) · [AI judgments and adjudications](ai-judgments.json) · [Physical request receipts](request-usage.jsonl)

## Final development matrix

| Variant | Executions | AI scores | Key-point correctness | Citation support | Citation completeness | First-round recall@5 | Median seconds | Chat attempts | Query embedding attempts | API estimate CNY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_llm | 5/5 | 5 | 0.483 | N/A | N/A | N/A | 5.775 | 5 | 0 | 0.0016218 |
| llm_rag | 5/5 | 5 | 0.900 | 0.880 | 0.770 | 1.000 | 4.860 | 5 | 5 | 0.0030234 |
| single_agent_rag | 5/5 | 5 | 0.900 | 0.850 | 0.812 | 0.833 | 8.117 | 10 | 24 | 0.0033833 |
| multi_agent_rag | 5/5 | 5 | 0.933 | 0.871 | 0.833 | 1.000 | 25.860 | 26 | 40 | 0.0195078 |

Correctness is a per-answer mean of reference-key-point scores (0/1/2), normalized to 0–1. Citation support checks entailment of attached factual claims; completeness includes uncited factual claims. They are macro means over applicable, valid judgments, not citation-ID validity. Applicable counts are explicit in summary.json; baseline citations and empty-gold recall are N/A. Failed execution and judge errors remain missing for semantic quality, with execution failures separately visible. A valid ID alone never establishes claim support. First-round recall uses the original global top five.

Shared index setup: 14 embedding requests, 0.014885 CNY estimated; 36914 ms. This setup is excluded from answer latency and per-variant cost. Request counts include failed attempts and bounded repairs. Trace model_calls counts both chat and query embeddings; the table separates them.

## Retained tuning history

| Phase | Successful traces | Failed traces | Chat attempts | Embedding attempts including setup | API estimate CNY |
| --- | ---: | ---: | ---: | ---: | ---: |
| [initial](initial-traces.jsonl) | 16 | 4 | 54 | 77 | 0.0594183 |
| [diagnostic](diagnostic-traces.jsonl) | 2 | 2 | 16 | 26 | 0.0312196 |
| [followup](followup-traces.jsonl) | 17 | 3 | 36 | 69 | unknown |
| [calibrated](calibrated-traces.jsonl) | 17 | 3 | 48 | 77 | 0.0538444 |
| [final](final-traces.jsonl) | 20 | 0 | 46 | 83 | 0.0424213 |

Initial failures exposed misspelled JSON fields, schema confusion, output truncation and missing inline citations. Follow-ups exposed long-ID transcription errors and a bounded citation repair that could repeat the same uncited body. The final code uses strict field output, a Markdown-first report schema, reversible request-local evidence labels for every RAG variant, and deterministic validation. Unknown labels and unsupported claims are not repaired by guessing. Invalid researcher/writer responses retain their paid usage even when the draft is rejected. Early missing trace counters remain visible as unassigned non-embedding requests in the physical ledger; they are not charged to index setup.

One embedding timeout during the historical followup phase has unknown usage and charge. Its reservation remains open; it is not treated as free. Thus that phase's total is unknown. The final generation run's request usage is complete.

## Answer and judge calibration findings

The closed-book final answers invented a generator-only cache-bypass requirement, incorrectly preferred a subclass exception handler, and fabricated deployment latency/benchmark figures. RAG answers avoided those particular errors, but some still incorrectly claimed that serialization_alias controls editor initializer signatures. The multi-agent insufficiency answer exposed internal critic commentary and used two extra retrieval loops without finding measurements. These are development findings to address, not claims of general superiority.

The initial judge returned 9 valid and 11 invalid judgments. Citation status was often confused with factual correctness, several claim spans changed Markdown/indentation, and long adjacent citation groups triggered a faulty character-window check. The binding validator now accepts a contiguous group after its first nearby token and rejects borrowing across paragraphs; entailment still requires source review. Chinese error-message translations and a documented alias+populate_by_name alternative were wrongly penalized. Every raw output and Codex decision is preserved; judge errors were never application-quality zeros.

Reference caveat: question 019 asks which exception classes to raise/handle, but its key also expects JSON-detail and an example import alias. The original additional criteria are retained in these scores and can penalize an otherwise relevant answer. Question 007 admits valid alternatives to its reference implementation. These pilot findings need reference clarification before full development-set scoring; the test set has not been tuned against model results.

## Reproduction and limits

Generation commit: `c53b1715909692669fcd1d5e6c31891faf496bf8`; prompt `evaluation-v5`. Chat snapshot: `qwen3.7-flash-2026-07-15` for all four variants and all chat roles, disabled thinking, temperature 0, cap 4096, 2 provider retries, at most 2 critic loops and 1 writer citation repair. Embedding: `text-embedding-v4`, 1024 dimensions; chunks 1200/200, top-k 5, whole eight-document development pool. Selection, exact serial execution order, seed, source/data hashes, raw judge responses and corrections are published. No query/answer cache supplies independent repeats.

DeepSeek Flash non-thinking scoring uses the official dsh CLI in account mode. System/model labels, costs and timings are excluded from judge input; answer style can still reveal architecture. AI reference review, AI scoring and Codex source checks can share biases and do not constitute human validation. The judge uses a rolling model alias, not a pinned snapshot. Sample size five does not support significance tests, admissions predictions, or general RAG-quality claims. Public-document pretraining contamination is possible. Latency is exploratory on a shared development machine.

Judge usage: 20 physical requests; 70908 input and 16377 output tokens (14592 cached input tokens, already included in input). Account pricing is unknown, so judge cost and the combined monetary total are unknown. DashScope costs are dated list-price estimates, not invoices. Six separate connectivity/constraint probes and earlier dataset-authoring/review calls are outside this dedicated pilot ledger.

Recompute published evidence without API calls:

```powershell
.venv/Scripts/python.exe scripts/verify_dev_pilot.py
```

Regenerate publication from retained local runtime records:

```powershell
.venv/Scripts/python.exe scripts/publish_dev_pilot.py --adjudications docs/evaluation/dev-pilot/adjudications.json
```

Next: complete the full 20-question development set, then freeze the experiment before running the repeated held-out comparison. This pilot does not freeze or complete that comparison.
