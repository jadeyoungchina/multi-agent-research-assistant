# Experiments and reproducibility

The versioned [dataset](../benchmarks/cases.jsonl) contains 30 synthetic cases
over six fictional [source files](../benchmarks/corpus). Counts describe fixture
structure, not measured model quality. Every case names its sources, expected
evidence phrases, answer key points, a loop bound, and a latency budget.
The repository verifier validates all cases and phrases against those files.
Corpus facts (including governance policies) are invented test material.

## Run the benchmark

After installing dependencies, from the repository root:

```bash
python -m app.evaluation.cli --provider fake --variants all --output-dir artifacts/evaluation/fake --enforce-gate
```

The complete case/variant matrix has 30 × 4 = 120 traces. This is the expected
matrix size; inspect generated `results.json` for actual completed/failed traces.
The command writes `results.json`, `results.csv`, and `report.md` to that output
directory. The files are generated evidence, deliberately excluded from Git.
CI uploads the same directory as the `fake-benchmark` artifact.

To run a real provider, configure your local `.env` as described in the README:

```bash
python -m app.evaluation.cli --provider dashscope --variants all --output-dir artifacts/evaluation/dashscope
python -m app.evaluation.cli --provider openai --variants all --output-dir artifacts/evaluation/openai
```

Real calls require account access and can incur charges. No saved real-provider
run is cited by this release documentation, so it makes no claim about Qwen or
OpenAI answer quality, speed, cost, or multi-agent improvement. A live smoke
test, when separately recorded, establishes connectivity for its configuration;
it does not establish benchmark quality. Save separate timestamped output
directories for repeated real runs before drawing any comparison.

## Four implementations

| Variant | Implemented sequence |
| --- | --- |
| `baseline_llm` | One direct chat answer; no corpus retrieval |
| `llm_rag` | Question retrieval, grounded structured answer, citation ID check |
| `single_agent_rag` | Query planning, expanded retrieval, grounded answer, citation ID check |
| `multi_agent_rag` | Product ResearchService with Planner, Retriever, Researcher, bounded Critic, Writer and citation validation |

The workflow interface receives only case ID, question, and source filenames.
Gold evidence/answer targets remain in the evaluator. Each source is ingested
once per benchmark run into disposable storage; application uploads/database
are not used. Latency and token accounting exclude that shared ingestion.

## What the metrics mean

Definitions are implemented in `app/evaluation/metrics.py` and covered by
`tests/evaluation/test_metrics.py`:

- `retrieval_recall_at_5`: fraction of expected source/phrase pairs present in
  the first five retrieved snapshots **after filtering to expected sources**.
  This source-filtered diagnostic differs from conventional global recall@5
  and can hide off-source ranking errors; interpret it with that limitation.
- `citation_precision`: unique cited IDs found among retrieved IDs divided by
  unique cited IDs. It is an identity check, not an entailment score.
- `evidence_coverage`: fraction of expected source/phrase pairs covered by cited evidence.
- `answer_key_point_f1`: mean best sentence token F1 for each reference point.
  Tokenization uses Latin words/numbers and individual CJK characters.
- `answer_key_point_coverage`: fraction of key points whose best sentence F1 is at least 0.60.
- Success/failure rates describe workflow execution. Insufficient evidence can
  still produce a successful trace with `evidence_sufficient=false`.
- p50/p95 use nearest-rank percentiles over externally measured milliseconds.
  Failed runs remain in aggregates and receive zero quality scores.
- Token totals, model-call counts, retries, and Critic loops are reported per
  trace and in aggregates. Calls include query embedding operations; fake token
  counts are zero because no language model is called, not an estimate of real cost.

Case loop and latency budgets are stored metadata; the current release gate
does not enforce these per-case budgets. Latency is machine-dependent and the
fake benchmark cannot measure real service performance. Fake embedding's
similarity threshold is set to -1.0 in the CLI to exercise deterministic evidence
flow, while the product default is 0.15. Fake and real scores are not comparable.

## Gate and trace provenance

Only `multi_agent_rag` blocks the v1.0 gate. These are configured acceptance
thresholds from [quality-gates.json](../benchmarks/quality-gates.json), not observed results:

| Metric | Required bound |
| --- | ---: |
| Success rate | ≥ 1.0 |
| Mean retrieval recall@5 | ≥ 0.9 |
| Mean citation precision | ≥ 1.0 |
| Mean evidence coverage | ≥ 0.9 |
| Mean answer key-point coverage | ≥ 0.8 |
| Failure rate | ≤ 0.0 |

CLI exit codes are 0 for completion/pass, 2 for invalid dataset, 3 for gate
failure, and 4 for execution/configuration failure. Reports are saved before
gate evaluation, so a failed gate can still be inspected. To evaluate only
selected variants, pass comma-separated names via `--variants`; an enforced
gate fails when `multi_agent_rag` was not evaluated.

JSON metadata includes UTC run time, Git commit and dirty flag, prompt version,
safe model/settings fields, and corpus/dataset SHA-256 digests. Corpus hashing
uses sorted POSIX relative filenames and exact bytes, each length-prefixed with
an unsigned 8-byte big-endian integer. Preserve those inputs and the report for
reproduction. Timing and generated document IDs can vary between fake runs.
Generated answers and evidence excerpts may contain private source text even
though configuration excludes keys and base URLs.
