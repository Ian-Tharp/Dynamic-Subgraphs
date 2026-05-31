# tests/test_api_jobs.py
from __future__ import annotations

import time

import pytest

from app.api.jobs import Job, JobExistsError, JobState, JobStore


def test_create_then_get() -> None:
    store = JobStore(max_workers=2)
    job = store.create("j1", kind="run")
    assert store.get("j1") is job
    assert job.state == JobState.QUEUED
    store.shutdown()


def test_duplicate_create_raises() -> None:
    store = JobStore(max_workers=1)
    store.create("dup", kind="run")
    with pytest.raises(JobExistsError):
        store.create("dup", kind="run")
    store.shutdown()


def test_submit_runs_to_completion() -> None:
    store = JobStore(max_workers=2)
    job = store.create("j2", kind="run")

    def work(j: Job) -> None:
        j.set_state(JobState.RUNNING)
        j.complete(result={"value": 42}, state=JobState.OK)

    store.submit(job, work)
    assert job.wait(timeout=2.0) is True
    assert job.state == JobState.OK
    assert job.result == {"value": 42}
    store.shutdown()


def test_exception_in_work_marks_failed() -> None:
    store = JobStore(max_workers=1)
    job = store.create("j3", kind="run")

    def boom(j: Job) -> None:
        raise RuntimeError("kaboom")

    store.submit(job, boom)
    assert job.wait(timeout=2.0) is True
    assert job.state == JobState.FAILED
    assert "kaboom" in (job.error or "")
    store.shutdown()


def test_wait_times_out_while_running() -> None:
    store = JobStore(max_workers=1)
    job = store.create("j4", kind="run")

    def slow(j: Job) -> None:
        j.set_state(JobState.RUNNING)
        time.sleep(0.5)
        j.complete(result=None, state=JobState.OK)

    store.submit(job, slow)
    assert job.wait(timeout=0.05) is False
    assert job.wait(timeout=2.0) is True
    store.shutdown()


def test_subscribe_yields_terminal_status() -> None:
    store = JobStore(max_workers=1)
    job = store.create("j5", kind="run")

    def work(j: Job) -> None:
        j.set_state(JobState.RUNNING)
        j.complete(result=None, state=JobState.OK)

    store.submit(job, work)
    job.wait(timeout=2.0)

    queue = job.subscribe()
    seen = []
    while True:
        msg = queue.get(timeout=2.0)
        if msg["type"] == "__end__":
            break
        seen.append(msg)
    states = [m["state"] for m in seen if m["type"] == "status"]
    assert "ok" in states
    store.shutdown()
