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

This first implementation adds the ledger and its concurrency/failure tests.
Provider and CLI integration are separate subsequent changes. Merely creating a
ledger does not constrain a caller that bypasses it.
