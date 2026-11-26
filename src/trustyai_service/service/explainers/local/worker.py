"""Run blocking local-explainer work without losing worker cleanup."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, cast

from .error_mapping import LocalWorkerCapacityError

if TYPE_CHECKING:
    from collections.abc import Callable

_MAX_CONCURRENT_WORKERS = 4
_WORKER_SLOTS = threading.BoundedSemaphore(_MAX_CONCURRENT_WORKERS)


async def run_local_worker[T](function: Callable[[], T], duration: float) -> T:
    """Run a synchronous explanation in a thread with a bounded await.

    Cancelling or timing out this coroutine does not cancel the underlying
    thread. The shield lets that thread finish its provider cleanup, while the
    callback consumes a late exception so it cannot become an unobserved task
    failure. The admission slot remains held until that thread actually
    finishes, including after the caller's timeout or cancellation.
    """
    if not _WORKER_SLOTS.acquire(blocking=False):
        raise LocalWorkerCapacityError

    try:
        task = asyncio.create_task(asyncio.to_thread(function))

        async def bridge() -> tuple[bool, T | BaseException]:
            """Observe the worker result while converting late errors to data."""
            try:
                return True, await task
            except BaseException as error:  # noqa: BLE001
                return False, error
            finally:
                _WORKER_SLOTS.release()

        bridge_task = asyncio.create_task(bridge())
    except BaseException:
        _WORKER_SLOTS.release()
        raise
    succeeded, result = await asyncio.wait_for(asyncio.shield(bridge_task), duration)
    if succeeded:
        return cast("T", result)
    raise cast("BaseException", result)
