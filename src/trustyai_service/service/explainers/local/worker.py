"""Bounded worker execution for CPU-bound local explanations."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_MAX_ACTIVE_WORKERS = 4
_WORKER_SLOTS = asyncio.Semaphore(_MAX_ACTIVE_WORKERS)


async def run_local_worker[T](function: Callable[[], T], duration: float) -> T:
    """Run one explanation in a bounded worker pool with a hard await deadline.

    Cancelling an asyncio thread task cannot stop Python code already running in
    the worker. Shielding it lets the worker finish its own provider cleanup;
    the slot is held until that cleanup completes, preventing timed-out requests
    from accumulating unbounded active work.
    """
    await _WORKER_SLOTS.acquire()
    task = asyncio.create_task(asyncio.to_thread(function))

    def release(_completed: asyncio.Future[object]) -> None:
        with suppress(asyncio.CancelledError):
            _completed.exception()
        _WORKER_SLOTS.release()

    try:
        return await asyncio.wait_for(asyncio.shield(task), duration)
    except BaseException:
        if task.done():
            _WORKER_SLOTS.release()
        else:
            task.add_done_callback(release)
        raise
    else:
        _WORKER_SLOTS.release()
