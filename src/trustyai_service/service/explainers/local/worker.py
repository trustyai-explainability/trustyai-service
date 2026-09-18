"""Bounded worker execution for CPU-bound local explanations."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


async def run_local_worker[T](function: Callable[[], T], duration: float) -> T:
    """Run one explanation in a thread with a hard await deadline.

    Cancelling an asyncio thread task cannot stop Python code already running in
    the worker. Shielding it lets the worker finish its own provider cleanup;
    the task callback consumes any late exception after the request has timed
    out.
    """
    task = asyncio.create_task(asyncio.to_thread(function))

    def consume(_completed: asyncio.Future[object]) -> None:
        with suppress(asyncio.CancelledError):
            _completed.exception()

    try:
        return await asyncio.wait_for(asyncio.shield(task), duration)
    except BaseException:
        if task.done():
            consume(task)
        else:
            task.add_done_callback(consume)
        raise
