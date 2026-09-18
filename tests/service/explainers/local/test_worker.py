"""Tests for local explainer worker deadlines and cleanup ownership."""

import asyncio
import threading

import pytest

from trustyai_service.service.explainers.local.worker import run_local_worker


@pytest.mark.asyncio
async def test_worker_returns_thread_result() -> None:
    """Return a completed synchronous worker result."""
    assert await run_local_worker(lambda: 42, 1) == 42


@pytest.mark.asyncio
async def test_worker_timeout_leaves_cleanup_to_the_worker() -> None:
    """A timeout does not cancel the thread that owns provider cleanup."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def work() -> str:
        started.set()
        release.wait(timeout=1)
        finished.set()
        return "done"

    task = asyncio.create_task(run_local_worker(work, 0.05))
    await asyncio.to_thread(started.wait, 1)
    with pytest.raises(TimeoutError):
        await task
    release.set()
    assert await asyncio.to_thread(finished.wait, 1)
