"""Process-pool isolation for native goodpoints MMD computation.

goodpoints has known heap-corruption bugs (SIGSEGV / free(): invalid size)
in its native (Cython) extensions. Running the computation in a subprocess
ensures a crash there kills only that worker, not the whole service.
"""

import asyncio
import functools
import logging
import multiprocessing
import os
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from trustyai_service.core.metrics.drift._goodpoints_patches import (
    apply_goodpoints_patches,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = 2

_state: dict[str, ProcessPoolExecutor | None] = {"pool": None}
_lock = threading.Lock()


def _max_workers() -> int:
    raw = os.getenv("TRUSTYAI_MMD_MAX_WORKERS", str(_DEFAULT_MAX_WORKERS))
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid TRUSTYAI_MMD_MAX_WORKERS=%r; using default %d",
            raw,
            _DEFAULT_MAX_WORKERS,
        )
        return _DEFAULT_MAX_WORKERS
    if value < 1:
        logger.warning(
            "TRUSTYAI_MMD_MAX_WORKERS must be >= 1, got %d; using default %d",
            value,
            _DEFAULT_MAX_WORKERS,
        )
        return _DEFAULT_MAX_WORKERS
    return value


def start_mmd_executor() -> None:
    """Create the shared MMD process pool. Call once from app lifespan startup."""
    with _lock:
        if _state["pool"] is None:
            # Use "spawn" (not the platform default "fork") since this process
            # is multi-threaded (asyncio event loop, background tasks) --
            # fork()-ing a multi-threaded process risks deadlocks on locks
            # held by other threads at fork time.
            max_workers = _max_workers()
            ctx = multiprocessing.get_context("spawn")
            _state["pool"] = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
                initializer=apply_goodpoints_patches,
            )
            logger.info("Started MMD process pool (max_workers=%d)", max_workers)


def shutdown_mmd_executor() -> None:
    """Shut down the shared MMD process pool. Call once from app lifespan shutdown."""
    with _lock:
        pool = _state["pool"]
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
            _state["pool"] = None
            logger.info("Shut down MMD process pool")


def reset_mmd_executor() -> None:
    """Reset singleton instance (useful for testing)."""
    with _lock:
        _state["pool"] = None


async def run_in_mmd_executor[T](func: Callable[..., T], /, **kwargs: Any) -> T:  # noqa: ANN401
    """Run func(**kwargs) in the shared MMD process pool."""
    pool = _state["pool"]
    if pool is None:
        msg = "MMD process pool not started; start_mmd_executor() must run during app startup"
        raise RuntimeError(msg)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool, functools.partial(func, **kwargs))
