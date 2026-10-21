**Stage B: request accounting and DeepSeek CLI integration · 2026-10-18**

Implemented persistent usage tracking, provider integration, incremental
evaluation recovery and a text-only DeepSeek CLI supervisor. The earlier CNY 100
ceiling was removed at the user's request; monetary enforcement remains opt-in.
Output limits, deadlines and bounded retries are operational controls.

Changes:

- Chat, embedding and CLI physical requests share a durable SQLite ledger.
  Usage is recorded before answer parsing; invalid JSON still counts. Partial
  usage keeps the known fields. Unknown prices and unknown usage are explicit.
- SDK retries are disabled; the application owns retry decisions. Authentication
  failures stop immediately. Truncated output fails without raising its limit.
- DeepSeek chat and embedding credentials are independent. DashScope and
  DeepSeek explicitly disable thinking for both plain and structured requests.
- Real evaluation runs default to five cases and write an incremental journal,
  a safe manifest and usage summary. Resume reuses completed units with their
  original timing and checks code/data/model/configuration identity. Corpus
  ingestion still runs again and its calls remain recorded.
- CLI account mode uses existing local authentication. A private overlay and
  request preloader restrict each helper to its declared model, request count
  and output limit. Completed helper IDs are not billed again on resume.

Validation:

- Full offline regression: **531 passed, 1 live test deselected**, 68.53 seconds.
- Four-workflow fake evaluation: 120 traces, release quality gate PASS.
- Actual installed `dsh` API-key and account profiles: mocked-provider SSE tests
  completed with `OFFLINE_MOCK_OK`; no real inference in those smoke checks.
- Real account CLI review: completed one request using `deepseek-flash`, with
  4,448 input tokens, 743 output tokens and zero reported cache-read tokens.
  Account price was not supplied, so no invoice amount is claimed.

The real review received only the ledger and request-accounting source. It
identified loss of known counters when a response supplies partial usage, and
counter updates occurring after persistence. Both were addressed and given
regression coverage. Suggested changes to idempotent unknown-state handling and
enforcement switching were not adopted as defects: preserving settled records
is intentional, and admission is checked inside the SQLite transaction.

The first provider-integration CI run exposed three tests that expected the old
metadata shape. Their assertions were updated for request kind, attempt count,
attempt IDs, usage completeness and cache tokens. No checks were removed or
disabled. Subsequent CI outcomes are linked in the development chat.

Local artifacts, prompts, raw model answers, CLI sessions and the usage database
remain outside tracked files. Published records contain the reviewed summary and
reproduction commands. This stage prepares execution infrastructure; the next
stage builds the licensed document set and corrects retrieval/citation metrics.

For physical-attempt accounting, the shared ledger is authoritative. Legacy
multi-agent graph counters still aggregate logical provider operations; the
next trace/metric revision will align those per-run counters with ledger rows.
