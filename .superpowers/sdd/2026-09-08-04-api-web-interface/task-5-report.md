# Task 5: Self-Contained Browser Dashboard

## Implementation details

- Added a same-origin `GET /` UI route and mounted local `/static` assets.
  Static-file resolution is derived from `app/main.py`, so a deployment can
  start the server outside the repository working directory.
- Added a dependency-free, semantic Chinese-language dashboard. It supports
  multi-file upload, document refresh/delete, source selection, research
  creation, progress display, retrying after failures, and a terminal report.
- The browser consumes only the existing `/api/documents`, `/api/research`,
  and research-event endpoints. No API key is rendered, requested, or stored.
- Progress uses one `EventSource` per active run. Starting another run closes
  the previous source; browser reconnects retain the SSE event cursor, while
  terminal `finished` and `failed` events close the active source and load the
  final REST result.
- All untrusted API/model data is added with explicit DOM nodes and
  `textContent`; the client contains no HTML-string injection operation. Report
  citations retain their evidence ID in `dataset.evidenceId`, and the modal
  looks up details only from the report's returned citation collection.
- A completed result with `evidence_sufficient === false` presents a visible
  `证据不足` warning before the report and retains all limitations. It is not
  treated as a request failure.
- Added system-font responsive styling, visible focus indicators,
  high-contrast error/warning states, cards, status chips, and the required
  single-column layout at 760px and below.

## Files changed

- `app/api/routes/ui.py` (new)
- `app/api/routes/__init__.py`
- `app/main.py`
- `app/static/index.html` (new)
- `app/static/styles.css` (new)
- `app/static/app.js` (new)
- `tests/static/conftest.py` (new)
- `tests/static/test_static_ui.py` (new)
- `.superpowers/sdd/2026-09-08-04-api-web-interface/task-5-report.md` (new)

## TDD evidence

The static dashboard contract was written before the route and asset files.
Its first focused run was RED: `GET /` returned 404 and the static files did
not exist. The test fixture was then placed in the static test directory so
the route test could exercise the application like the API suite.

### Initial RED command

```powershell
.venv\Scripts\python.exe -m pytest tests/static/test_static_ui.py -v
```

### Initial RED result

```text
FAILED test_index_contains_complete_research_controls - assert 404 == 200
FAILED test_frontend_is_self_contained_and_avoids_html_injection - FileNotFoundError
FAILED test_dashboard_assets_cover_safe_research_lifecycle - FileNotFoundError
```

This established the missing route and missing assets before production UI
code was introduced.

### Static-directory robustness RED command

```powershell
.venv\Scripts\python.exe -m pytest tests/static/test_static_ui.py::test_static_assets_are_served_when_the_server_starts_elsewhere -v
```

### Static-directory robustness RED result

```text
RuntimeError: Directory 'app/static' does not exist
```

The test changed the process working directory before creating the app. The
static mount now uses an absolute package-relative directory.

### Focused GREEN verification

```powershell
.venv\Scripts\python.exe -m pytest tests/static/test_static_ui.py -v
node --check app/static/app.js
.venv\Scripts\python.exe -m pytest tests/static tests/api -q
```

```text
4 passed
JavaScript syntax check exited 0
65 passed
```

## Self-review

- The UI never uses `innerHTML`, remote URLs, CDN scripts, external fonts, or
  client-side API-key handling.
- Forms have labels; dynamically updated status, document, timeline, and
  report regions announce changes; controls have keyboard-visible focus.
- The dialogue receives focus when opened and has a dedicated close control.
- Network, parse, upload, delete, research-creation, stream, failed-run, and
  missing-citation paths surface user-visible messages instead of silent
  failures.
- `git diff --check` reported no whitespace errors.

## Concerns

The EventSource `error` callback intentionally leaves a non-terminal stream
open so the browser can perform its built-in cursor-based reconnect. It tells
the user that it is reconnecting; a terminal error is instead represented by a
workflow `failed` event and final REST status.
