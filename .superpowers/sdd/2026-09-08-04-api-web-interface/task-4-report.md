# Task 4: Resumable Server-Sent Events

## Implementation details

- Added `encode_sse()` to serialize each `RunEvent` as a UTF-8 SSE frame with
  `id`, `workflow` event type, compact Pydantic JSON data, and the required
  blank-line terminator.
- Added `stream_run_events()`, which reads only through the injected research
  service, resumes after a cursor, emits keep-alive comments while a run is
  active, stops promptly on disconnect, and closes after a completed or failed
  run has no remaining events.
- Added `GET /api/research/{run_id}/events`. It validates the run before
  returning a stream, takes the larger of `after` and a valid integer
  `Last-Event-ID`, and sets `Cache-Control: no-cache` plus
  `X-Accel-Buffering: no`.
- The route uses the existing service dependency and error middleware; it does
  not import agents, provider SDKs, or SQL.

## Files changed

- `app/api/sse.py` (new)
- `app/api/routes/research.py`
- `tests/api/test_sse.py` (new)
- `.superpowers/sdd/2026-09-08-04-api-web-interface/task-4-report.md` (new)

## TDD evidence

The behavior-focused tests were written before production SSE code. The RED
failure was expected because the new SSE module had not been created yet.

### RED command

```powershell
.venv\Scripts\python.exe -m pytest tests/api/test_sse.py -q
```

### RED output

```text
=================================== ERRORS ====================================
___________________ ERROR collecting tests/api/test_sse.py ____________________
ImportError while importing test module 'D:\AIProjects\multi-agent-research-assistant\.worktrees\rebuild-project\tests\api\test_sse.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Admin\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\api\test_sse.py:9: in <module>
    from app.api.sse import encode_sse
E   ModuleNotFoundError: No module named 'app.api.sse'
=========================== short test summary info ===========================
ERROR tests/api/test_sse.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.42s
```

It failed as expected: the test expressed the intended new public module and
encoder before either existed.

### GREEN command and output

```powershell
.venv\Scripts\python.exe -m pytest tests/api/test_sse.py -q
```

```text
.....                                                                    [100%]
5 passed in 0.31s
```

### Final verification command and output

```powershell
.venv\Scripts\python.exe -m pytest -q
```

```text
........................................................................ [ 24%]
........................................................................ [ 48%]
........................................................................ [ 72%]
........................................................................ [ 96%]
............                                                             [100%]
300 passed in 7.53s
```

## Self-review

- Resume precedence is covered with query `after=2` and `Last-Event-ID: 3`;
  only event 4 is delivered.
- The terminal fake run verifies backlog delivery then a clean close.
- Direct generator tests cover heartbeat emission and no service query after a
  disconnect.
- The missing-run route test confirms normal domain-error handling and the
  request-ID middleware header remain in effect.
- `git diff --check` produced no whitespace errors.

## Concerns

None. The polling and heartbeat values are deliberately local route constants
to keep the stream behavior explicit and avoid expanding the application
configuration surface.

## Fix round: terminal completion race

### Finding and fix

The SSE loop previously read events and run status separately. A final event
could be committed after an empty event read but before the terminal status
read, which made the loop close without emitting that event. The terminal
decision now rechecks events from the unchanged cursor when it observes a
completed or failed run after an empty read. It only closes when that second
read is also empty.

### Covering test file

`tests/api/test_sse.py`

`test_event_generator_rechecks_events_committed_during_terminal_transition`
uses a fake service that commits event 1 and transitions to `completed` in the
gap between the generator's first `list_events()` call and `get_run()` call.
It asserts that the stream still emits event 1.

### RED command

```powershell
.venv\Scripts\python.exe -m pytest tests/api/test_sse.py -q
```

### RED output

```text
.....F                                                                   [100%]
================================== FAILURES ===================================
__ test_event_generator_rechecks_events_committed_during_terminal_transition __

    def test_event_generator_rechecks_events_committed_during_terminal_transition() -> None:
        class CompletionRaceService(FakeResearchService):
            def __init__(self) -> None:
                super().__init__()
                self.events = []
                self.run = _run(status="running")
                self.list_calls = 0

            def list_events(
                self, run_id: str, after_sequence: int = 0
            ) -> list[RunEvent]:
                self.list_calls += 1
                return [event for event in self.events if event.sequence > after_sequence]

            def get_run(self, run_id: str) -> ResearchRun:
                if self.list_calls == 1:
                    self.events.append(_event(1))
                    self.run = _run(status="completed")
                return super().get_run(run_id)

        async def read_events() -> list[bytes]:
            return [
                event
                async for event in stream_run_events(
                    RequestThatStaysConnected(), "run1", CompletionRaceService(), 0, 0, 60
                )
            ]

>       assert asyncio.run(read_events()) == [encode_sse(_event(1))]
E       assert [] == [b'id: 1\neve...00:00Z"}\n\n']
E
E         Right contains one more item: b'id: 1\nevent: workflow\ndata: {"sequence":1,"run_id":"run1","stage":"planner","event_type":"created","payload":{"question":"What does the evidence show?"},"created_at":"2026-09-10T00:00:00Z"}\n\n'
E         Use -v to get more diff

tests\api\test_sse.py:187: AssertionError
=========================== short test summary info ===========================
FAILED tests/api/test_sse.py::test_event_generator_rechecks_events_committed_during_terminal_transition
1 failed, 5 passed in 0.65s
```

The failure reproduces the reported race: the stream ended with no frame even
though the terminal event was committed before it closed.

### GREEN command and output

```powershell
.venv\Scripts\python.exe -m pytest tests/api/test_sse.py -q
```

```text
......                                                                   [100%]
6 passed in 0.30s
```

### Final focused and full verification

```powershell
.venv\Scripts\python.exe -m pytest tests/api/test_sse.py -q
```

```text
......                                                                   [100%]
6 passed in 0.30s
```

```powershell
.venv\Scripts\python.exe -m pytest -q
```

```text
........................................................................ [ 23%]
........................................................................ [ 47%]
........................................................................ [ 71%]
........................................................................ [ 95%]
.............                                                            [100%]
301 passed in 7.53s
```
