**Tracked DeepSeek CLI helper tasks**

The helper supervisor runs the installed official `dsh` CLI in account-login
mode by default. It uses a private per-task patch and a temporary workspace,
leaving the user's saved CLI configuration intact. No API key is required for
the account route. The current commands record usage without a monetary limit.

Write a small, explicit task and its necessary source excerpts to a UTF-8 file
under `artifacts/`, then run from the repository root:

```powershell
.venv/Scripts/python.exe -m app.evaluation.deepseek_cli --prompt-file artifacts/tasks/review.txt --task-id review-001 --output artifacts/tasks/review-001.json
```

This command calls DeepSeek using the existing local CLI account authentication.
It runs one text-in/text-out task, defaults to one inference request and 512
output tokens, and disables thinking. For a longer review, add
`--max-output-tokens 1536`. There is a supervisor deadline and no automatic
provider retries. A task may declare up to three requests with `--max-requests`.

Completed task IDs return the saved result without launching the CLI again.
Reusing an ID with changed input/configuration is refused. Failed or interrupted
tasks retain their recorded attempts and require a new task ID after review;
they are never silently retried. Each independent experiment repetition must
have its own identity. The ledger defaults to
`artifacts/evaluation/budget/ledger.sqlite`.

The Node preloader intercepts the supported CLI's inference fetches and admits
each through a local tracking service. It fixes the model to `deepseek-flash`,
removes tool declarations and session-log request extensions, disables thinking,
and records response usage before returning output to the CLI. Files, images,
tool calls, arbitrary destinations and hidden additional requests are refused.
The local tracking service receives no authentication headers. Known tool
providers are also disabled in the per-task patch. This is an integration for
the trusted, tested CLI version, rather than an operating-system sandbox for
arbitrary third-party plugins.

Account prices remain unspecified until independently verified. The ledger
records account input/output/cache tokens with `state=unpriced` and
`total_cost_cny=null`; zero is not substituted for an unknown invoice amount.
Partial or missing usage remains explicitly incomplete. Optional monetary
enforcement requires `--enforce-budget` and a verified, current CNY account-price
record supplied with `--account-price-file`. API-key billing is a separate route
selected by `--mode api_key`; it uses `DEEPSEEK_API_KEY` from the local project
configuration or launch environment.

Validated against `@deepseek-ai/dsh` 0.2.1-rc.2. Node subprocess and local service
tests use mocked provider traffic; both actual installed CLI profiles were also
checked with simulated SSE responses. A subsequent real account-mode code review
completed with one inference request, 4,448 input tokens and 743 output tokens.
The [stage-B record](step-b-implementation.md) describes that validation and the
review findings. It validates CLI transport and usage recording.
