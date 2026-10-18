# Five-question development pilot

Selection recorded before running the model, 2026-10-18. This is an engineering
and scoring calibration, not a held-out quality estimate. Reference answers have
AI review; human review remains pending. The 40 test questions will not be run.

| Case | Type | Language | Family |
| --- | --- | --- | --- |
| REAL-DEV-001 | Single-document fact | Chinese | Pydantic coercion |
| REAL-DEV-007 | Cross-document synthesis | English | Pydantic fields and aliases |
| REAL-DEV-013 | Conditions, limits and numbers | Chinese | FastAPI dependencies |
| REAL-DEV-019 | Version and apparent conflict | English | FastAPI errors |
| REAL-DEV-020 | Insufficient corpus evidence | English | FastAPI errors |

All RAG variants search the same eight-document development pool, independent of
gold sources. Serial execution uses seed `20261018`: shuffle case blocks and a
base variant permutation, then rotate variant positions across blocks. The first
four blocks balance each variant across all four positions; the fifth block is
necessarily unbalanced. Exact execution order is saved in the run manifest and
report. This controls position, not all temporal or service-cache effects.

Use the same Qwen snapshot `qwen3.7-flash-2026-07-15` for every chat stage, disabled
thinking, temperature 0, output cap 2048, two application retries, and DashScope
`text-embedding-v4` (1024 dimensions). Both models passed an accounted live
connectivity check. Index construction is excluded from per-answer time; its
requests and usage will be reported separately. A dedicated pilot ledger avoids
mixing prior dataset-authoring costs into experiment costs.

```powershell
.venv/Scripts/python.exe -m app.evaluation.cli --provider dashscope --embedding-provider dashscope --chat-model qwen3.7-flash-2026-07-15 --real-manifest benchmarks/real-v1/corpus-manifest.json --split dev --allow-ai-reviewed --case-ids REAL-DEV-001,REAL-DEV-007,REAL-DEV-013,REAL-DEV-019,REAL-DEV-020 --order-seed 20261018 --ledger artifacts/evaluation/real-v1-pilot/ledger.sqlite --output-dir artifacts/evaluation/real-v1-pilot/initial
```

Keep failed traces and costs. Any changed configuration runs in a new directory;
do not overwrite the initial pilot. DeepSeek's official CLI will assess blinded
answers against the semantic rubric, followed by Codex source checks. Scoring
errors remain missing until corrected and adjudications are recorded explicitly.
AI scores must not be presented as human judgments or admission predictions.

After the [initial and diagnostic failures](dev-pilot-initial.md), repeat exactly
the same selection and seed in a fresh `followup` output directory, adding
`--structured-output json_schema --max-output-tokens 4096 --capture-responses`.
This is development tuning, not an independent test. Keep all earlier traces.
