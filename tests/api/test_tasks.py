import logging
import threading

from app.api.tasks import RunTaskManager


class BlockingResearchService:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()
        self.execution_count = 0
        self.executed_run_ids: list[str] = []

    def execute(self, run_id: str) -> str:
        self.execution_count += 1
        self.executed_run_ids.append(run_id)
        self.started.set()
        self.release.wait(timeout=2)
        self.completed.set()
        return run_id


class RecordingResearchService:
    def __init__(self) -> None:
        self.statuses: dict[str, str] = {}
        self.completed = threading.Event()

    def execute(self, run_id: str) -> str:
        if run_id == "failed-run":
            self.statuses[run_id] = "failed"
            raise RuntimeError("sentinel provider response")
        self.statuses[run_id] = "completed"
        self.completed.set()
        return run_id


def test_task_manager_deduplicates_active_run() -> None:
    service = BlockingResearchService()
    manager = RunTaskManager(service, max_workers=1)

    manager.start("run1")
    assert service.started.wait(timeout=2)
    manager.start("run1")
    service.release.set()
    assert service.completed.wait(timeout=2)
    manager.close()

    assert service.execution_count == 1


def test_close_waits_for_running_work_and_cancels_only_queued_futures(
    monkeypatch,
) -> None:
    service = BlockingResearchService()
    manager = RunTaskManager(service, max_workers=1)
    manager.start("running")
    assert service.started.wait(timeout=2)
    manager.start("queued")
    shutdown_started = threading.Event()
    original_shutdown = manager.executor.shutdown

    def recording_shutdown(*args, **kwargs) -> None:
        shutdown_started.set()
        original_shutdown(*args, **kwargs)

    monkeypatch.setattr(manager.executor, "shutdown", recording_shutdown)

    close_thread = threading.Thread(target=manager.close)
    close_thread.start()

    assert shutdown_started.wait(timeout=2)
    assert close_thread.is_alive()
    assert not service.completed.is_set()

    service.release.set()
    close_thread.join(timeout=2)

    assert not close_thread.is_alive()
    assert service.executed_run_ids == ["running"]
    assert service.completed.is_set()


def test_failed_run_does_not_prevent_subsequent_run_from_completing() -> None:
    service = RecordingResearchService()
    manager = RunTaskManager(service, max_workers=1)

    manager.start("failed-run")
    manager.start("completed-run")
    assert service.completed.wait(timeout=2)
    manager.close()

    assert service.statuses == {
        "failed-run": "failed",
        "completed-run": "completed",
    }


def test_failed_task_log_contains_only_safe_context(caplog) -> None:
    service = RecordingResearchService()
    manager = RunTaskManager(service, max_workers=1)

    with caplog.at_level(logging.ERROR, logger="app.api.tasks"):
        manager.start("failed-run")
        manager.close()

    record = next(
        record for record in caplog.records if record.msg == "research task failed"
    )
    assert record.run_id == "failed-run"
    assert record.error_type == "RuntimeError"
    assert "sentinel provider response" not in caplog.text
