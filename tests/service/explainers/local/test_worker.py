"""Tests for cancellation-safe local-explainer worker execution."""

from __future__ import annotations

import asyncio
import threading

import pytest

from trustyai_service.service.explainers.local import worker as worker_module
from trustyai_service.service.explainers.local.worker import run_local_worker


@pytest.mark.asyncio
async def test_worker_runs_sync_function_in_a_thread() -> None:
    """Return the result of a synchronous worker function."""
    assert await run_local_worker(lambda: 42, 1.0) == 42


@pytest.mark.asyncio
async def test_timed_out_worker_holds_bounded_capacity_until_thread_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject new work while a timed-out thread still consumes its slot."""
    monkeypatch.setattr(
        worker_module,
        "_WORKER_SLOTS",
        threading.BoundedSemaphore(1),
        raising=False,
    )
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def work() -> str:
        started.set()
        release.wait(timeout=1)
        finished.set()
        return "done"

    first = asyncio.create_task(run_local_worker(work, 0.05))
    assert await asyncio.to_thread(started.wait, 1.0)
    with pytest.raises(TimeoutError):
        await first

    with pytest.raises(RuntimeError, match="capacity"):
        await run_local_worker(lambda: 42, 1.0)

    release.set()
    assert await asyncio.to_thread(finished.wait, 1.0)
    await asyncio.sleep(0)
    assert await run_local_worker(lambda: 42, 1.0) == 42


@pytest.mark.asyncio
async def test_worker_timeout_shields_cleanup_and_consumes_late_exception() -> None:
    """Keep worker-owned cleanup running after timeout and consume its error."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    loop.set_exception_handler(
        lambda _loop, context: loop_errors.append(context),
    )

    def work() -> None:
        started.set()
        release.wait(timeout=1)
        finished.set()
        message = "late worker failure"
        raise RuntimeError(message)

    task = asyncio.create_task(run_local_worker(work, 0.05))
    assert await asyncio.to_thread(started.wait, 1.0)
    with pytest.raises(TimeoutError):
        await task

    release.set()
    assert await asyncio.to_thread(finished.wait, 1.0)
    await asyncio.sleep(0.05)
    assert loop_errors == []


@pytest.mark.asyncio
async def test_worker_cancellation_does_not_cancel_thread_or_close_provider() -> None:
    """Cancellation leaves the synchronous function responsible for cleanup."""
    started = threading.Event()
    release = threading.Event()
    cleanup = threading.Event()

    def work() -> str:
        started.set()
        try:
            release.wait(timeout=1)
            return "done"
        finally:
            cleanup.set()

    task = asyncio.create_task(run_local_worker(work, 5.0))
    assert await asyncio.to_thread(started.wait, 1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not cleanup.is_set()

    release.set()
    assert await asyncio.to_thread(cleanup.wait, 1.0)
