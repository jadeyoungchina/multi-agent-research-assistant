**Shared request budget**

`app/evaluation/budget.py` implements a local SQLite ledger shared by independent
provider calls and CLI supervisors. Amounts are integer nano-CNY. Its initial
policy is persisted: reopening the same file cannot silently increase the limit.
The default total is CNY 100, with admission stopping at CNY 90 to retain margin
for uncertain charges. A lower budget can be selected when creating a ledger.

Before network I/O, reserve the full uncached input bound and maximum output.
The admission transaction includes all completed charges, in-flight reservations
and unresolved charges from previous processes. It does not expire reservations.
Each physical attempt has its own row. Record response usage before parsing the
answer so that invalid JSON still counts. A timeout with no trustworthy usage
retains its reservation. Actual usage exceeding the bound is recorded and stops
further admission, including when the overall cap has not yet been reached.

The ledger stores request hashes, provider/model/kind/role labels, price versions
and token counts. It never stores prompts or authentication headers. Keep the
database under `artifacts/`, which Git ignores. The summary distinguishes measured
usage charges from reserved or unknown amounts; neither is a substitute for the
provider's invoice. Settlement of an unknown attempt requires explicit reconciled
usage. Never release a reservation simply because a process exited.

Built-in rates are dated, conservative API-key prices from the
[DeepSeek pricing page](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)
and [Model Studio pricing page](https://help.aliyun.com/zh/model-studio/model-pricing).
Qwen rates apply to Beijing and requests within the declared input tier; requests
above the tier are refused. Recheck rates and account region before paid runs.
DeepSeek account-login billing is not assumed to equal API-key billing.

The evaluation CLI uses the ledger for both chat and embedding, including corpus
ingestion. Real-provider runs default to five cases. For example:

```powershell
.venv/Scripts/python.exe -m app.evaluation.cli --provider dashscope --embedding-provider dashscope --limit 5 --output-dir artifacts/evaluation/pilot-01
```

The command makes paid requests. Credentials must already be configured locally.
`--max-output-tokens` defaults to 2048; the SDK's own retries are disabled, and the
application owns bounded retries. Authentication/configuration failures do not
retry. Truncation fails without silently increasing the output limit. DashScope
and DeepSeek use explicit non-thinking mode for both plain and structured calls.
`--provider deepseek` uses a separately configured API key, not a CLI account;
its embedding provider defaults to DashScope. Unpriced providers or mismatched
endpoint regions cannot enter a budgeted run.

Real runs write an incremental `traces.jsonl`, a safe configuration manifest and
`budget-summary.json`, in addition to the usual reports. Budget stop returns code
5 and preserves the partial matrix, with `execution_complete=false`. Use the same
command with `--resume` to reuse recorded units and their original timing; source,
model, corpus, dataset and configuration changes are rejected. Each independent
experimental repeat must use a new output directory. Corpus ingestion currently
runs again on resume and remains accounted; persistent embedding caching is a
later change. Failed requests with unknown charges retain their reservations.

The ordinary application factory also disables SDK retries and caps output, but
the research web application does not attach this experimental shared ledger by
default. The ledger constrains callers that explicitly use it; merely creating a
database does not constrain a process that bypasses it.
