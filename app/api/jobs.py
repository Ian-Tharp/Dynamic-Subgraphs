# app/api/jobs.py
"""In-process job store: background execution + subscribe bus.

Every run/chain becomes a Job executed on a thread pool. The request handler
decides how long to wait (sync/async/auto). The subscribe bus feeds SSE today;
it is the seam where per-node events will publish later.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from queue import Queue
from typing import Any

_TERMINAL: frozenset[str] = frozenset({"ok", "failed", "paused"})


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    PAUSED = "paused"


class JobExistsError(Exception):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Job already exists: {run_id!r}")
        self.run_id = run_id


def _now() -> datetime:
    return datetime.now(UTC)


class Job:
    """Mutable, thread-safe handle for one background run/chain."""

    def __init__(self, run_id: str, kind: str) -> None:
        self.run_id = run_id
        self.kind = kind
        self.state: JobState = JobState.QUEUED
        self.submitted_at: datetime = _now()
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.result: Any = None
        self.error: str | None = None
        self.budget_wall_seconds: int | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._subscribers: list[Queue[dict[str, Any]]] = []

    def _publish(self, msg: dict[str, Any]) -> None:
        for q in self._subscribers:
            q.put(msg)

    def set_state(self, state: JobState) -> None:
        with self._lock:
            self.state = state
            if state == JobState.RUNNING and self.started_at is None:
                self.started_at = _now()
            self._publish({"type": "status", "state": state.value})

    def complete(self, *, result: Any, state: JobState) -> None:
        with self._lock:
            self.result = result
            self.state = state
            self.finished_at = _now()
            self._publish({"type": "status", "state": state.value})
            self._publish({"type": "__end__"})
            self._subscribers.clear()
            self._done.set()

    def fail(self, message: str) -> None:
        with self._lock:
            self.error = message
            self.state = JobState.FAILED
            self.finished_at = _now()
            self._publish({"type": "status", "state": JobState.FAILED.value})
            self._publish({"type": "__end__"})
            self._subscribers.clear()
            self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout=timeout)

    def is_terminal(self) -> bool:
        return self.state.value in _TERMINAL

    def subscribe(self) -> Queue[dict[str, Any]]:
        q: Queue[dict[str, Any]] = Queue()
        with self._lock:
            q.put({"type": "status", "state": self.state.value})
            if self.is_terminal():
                q.put({"type": "__end__"})
            else:
                self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Queue[dict[str, Any]]) -> None:
        """Drop a subscriber's queue so a disconnected client can't leak it.

        Idempotent: a queue already removed (by a terminal `clear()`) or never
        registered (a terminal-at-subscribe job) is silently ignored.
        """
        with self._lock, contextlib.suppress(ValueError):
            self._subscribers.remove(q)


class JobStore:
    def __init__(self, max_workers: int = 4) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ds-job"
        )

    def create(self, run_id: str, *, kind: str) -> Job:
        with self._lock:
            if run_id in self._jobs:
                raise JobExistsError(run_id)
            job = Job(run_id=run_id, kind=kind)
            self._jobs[run_id] = job
            return job

    def submit(self, job: Job, fn: Callable[[Job], None]) -> None:
        self._executor.submit(self._wrap, job, fn)

    @staticmethod
    def _wrap(job: Job, fn: Callable[[Job], None]) -> None:
        try:
            fn(job)
        except Exception as exc:
            job.fail(f"{type(exc).__name__}: {exc}")

    def get(self, run_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(run_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
