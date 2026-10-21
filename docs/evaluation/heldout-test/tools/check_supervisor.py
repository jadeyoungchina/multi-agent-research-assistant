"""Offline supervisor contracts: no network, providers, Git mutations or paid calls."""
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

import continue_run as worker


class Clock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Response(io.StringIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def runs(*, ready=True, failure=False, sha="abc"):
    return {"workflow_runs": [{"id": index, "head_sha": sha, "event": event, "name": "ci",
            "status": "completed" if ready else "queued",
            "conclusion": ("failure" if failure and index == 0 else "success") if ready else None,
            "html_url": "https://github.com/example/run"}
            for index, event in enumerate(("push", "pull_request"))]}


class SupervisorContracts(unittest.TestCase):
    def test_fifteen_minute_queue_reaches_both_successes_without_rate_pressure(self):
        clock, requests = Clock(), []

        def fetch(*args, **kwargs):
            requests.append(clock.now)
            return Response(json.dumps(runs(ready=clock.now >= 900)))

        with patch.object(worker.time, "monotonic", clock.monotonic), patch.object(worker.time, "sleep", clock.sleep), \
                patch.object(worker.urllib.request, "urlopen", side_effect=fetch):
            self.assertEqual(len(worker.wait_ci("abc")), 2)
        self.assertLessEqual(len(requests), 10)
        self.assertGreaterEqual(clock.now, 900)

    def test_completed_failure_still_stops_without_waiting(self):
        with patch.object(worker.urllib.request, "urlopen", return_value=Response(json.dumps(runs(failure=True)))), \
                patch.object(worker.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError):
                worker.wait_ci("abc")
            sleep.assert_not_called()

    def test_other_commit_does_not_satisfy_ci_gate(self):
        responses = [Response(json.dumps(runs(sha="wrong"))), Response(json.dumps(runs()))]
        with patch.object(worker.urllib.request, "urlopen", side_effect=responses), patch.object(worker.time, "sleep") as sleep:
            self.assertEqual(len(worker.wait_ci("abc")), 2)
            sleep.assert_called_once()

    def test_rate_limit_honors_retry_after_without_assuming_success(self):
        error = urllib.error.HTTPError("https://api.github.com", 403, "limited", {"Retry-After": "300"}, None)
        with patch.object(worker.urllib.request, "urlopen", side_effect=[error, Response(json.dumps(runs()))]), \
                patch.object(worker.time, "sleep") as sleep:
            self.assertEqual(len(worker.wait_ci("abc")), 2)
            self.assertGreaterEqual(sleep.call_args.args[0], 300)

    def test_perpetual_queue_eventually_stops_without_calling_it_success(self):
        clock = Clock()
        with patch.object(worker, "CI_MAX_WAIT_SECONDS", 360), patch.object(worker.time, "monotonic", clock.monotonic), \
                patch.object(worker.time, "sleep", clock.sleep), \
                patch.object(worker.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(json.dumps(runs(ready=False)))):
            with self.assertRaises(TimeoutError):
                worker.wait_ci("abc")
        self.assertEqual(clock.now, 360)

    def test_failed_resume_identity_never_starts_any_command(self):
        with patch.object(worker, "load_repeat", side_effect=ValueError("identity differs")), patch.object(worker, "command") as command:
            with self.assertRaises(ValueError):
                worker.validate_prior_repeats({"freeze_id": "abc"}, 2)
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
