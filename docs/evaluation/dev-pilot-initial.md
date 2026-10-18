# Initial live development pilot: failures retained

2026-10-18, generation commit `9848e10`. The prespecified five questions produced
20 fresh traces with the fixed Qwen snapshot and real DashScope embeddings.
16 succeeded; four failed. This calibration is not a held-out quality comparison.

| Variant | Execution success | Failures |
| --- | --- | --- |
| baseline_llm | 5/5 | None |
| llm_rag | 4/5 | Invalid structured response on REAL-DEV-007 |
| single_agent_rag | 5/5 | None |
| multi_agent_rag | 2/5 | Invalid structure (019), truncation (020), citation validation (001) |

A separate four-trace diagnostic run reproduced the structured-output failures.
Receipts show `answer_markmarkdown` / `answer_mark` instead of `answer_markdown`,
and a synthesis-shaped object / `report` instead of DraftReport's required fields.
These are schema failures, not evidence of incorrect reference answers.

The follow-up will opt into closed JSON Schema output, retain local generated-text
receipts, use the same 4096-token output cap for every variant, and ask the writer
for a compact report whose declared evidence IDs also appear inline. Plain-text
baseline remains plain text. Strict output is an explicit provider option because
other models may not support it. Both GroundedAnswer and DraftReport passed live
strict-mode probes. The schema format follows the [official DashScope documentation](https://help.aliyun.com/zh/model-studio/qwen-structured-output).

Writer citation failures previously lost the already-completed write call from
trace usage. Its metadata now propagates through the failed checkpoint; the
physical request ledger already retained that call. Initial trace counters must
therefore not be treated as a complete cost source. All experiment costs will be
reconciled to physical attempts, including failures and setup.

For the missing-evidence latency question (020), the initial closed-book answer
invented deployment-specific microsecond ranges. Both simpler RAG answers stated
that the corpus supplied no measurement. Multi-agent execution truncated, so its
answer quality is missing. No variant is declared the winner from this pilot.
Reference review is AI-only; human review and held-out evaluation remain pending.
