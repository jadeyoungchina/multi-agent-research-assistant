import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any


logger = logging.getLogger(__name__)


class RunTaskManager:
    def __init__(self, research_service: Any, max_workers: int = 2) -> None:
        self.service = research_service
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="research",
        )
        self.futures: dict[str, Future] = {}
        self.lock = Lock()
        self._closed = False

    def start(self, run_id: str) -> None:
        with self.lock:
            if self._closed:
                raise RuntimeError("task manager is closed")
            current = self.futures.get(run_id)
            if current is not None and not current.done():
                return
            future = self.executor.submit(self.service.execute, run_id)
            self.futures[run_id] = future
        future.add_done_callback(lambda completed: self._discard(run_id, completed))

    def _discard(self, run_id: str, completed: Future) -> None:
        with self.lock:
            if self.futures.get(run_id) is completed:
                self.futures.pop(run_id, None)
        if completed.cancelled():
            return
        exception = completed.exception()
        if exception is not None:
            logger.error(
                "research task failed",
                extra={
                    "run_id": run_id,
                    "error_type": type(exception).__name__,
                },
            )

    def wait(self, run_id: str, timeout: float | None = None) -> Any:
        with self.lock:
            future = self.futures.get(run_id)
        return None if future is None else future.result(timeout=timeout)

    def close(self) -> None:
        with self.lock:
            self._closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
