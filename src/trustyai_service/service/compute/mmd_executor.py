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
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from trustyai_service.core.metrics.drift._goodpoints_patches import (
    apply_goodpoints_patches,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = 2
_DEFAULT_TIMEOUT_SECONDS = 300.0

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


def _timeout_seconds() -> float:
    raw = os.getenv("TRUSTYAI_MMD_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid TRUSTYAI_MMD_TIMEOUT_SECONDS=%r; using default %s",
            raw,
            _DEFAULT_TIMEOUT_SECONDS,
        )
        return _DEFAULT_TIMEOUT_SECONDS
    if value <= 0:
        logger.warning(
            "TRUSTYAI_MMD_TIMEOUT_SECONDS must be > 0, got %s; using default %s",
            value,
            _DEFAULT_TIMEOUT_SECONDS,
        )
        return _DEFAULT_TIMEOUT_SECONDS
    return value


def _create_pool() -> ProcessPoolExecutor:
    # Use "spawn" (not the platform default "fork") since this process
    # is multi-threaded (asyncio event loop, background tasks) --
    # fork()-ing a multi-threaded process risks deadlocks on locks
    # held by other threads at fork time.
    max_workers = _max_workers()
    ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=apply_goodpoints_patches,
    )
    logger.info("Started MMD process pool (max_workers=%d)", max_workers)
    return pool


def start_mmd_executor() -> None:
    """Create the shared MMD process pool. Call once from app lifespan startup."""
    with _lock:
        if _state["pool"] is None:
            _state["pool"] = _create_pool()


def _replace_pool(stale_pool: ProcessPoolExecutor, *, reason: str) -> None:
    """Replace an unusable pool with a fresh one so future calls can succeed.

    Used both when a worker crashes (the pool is permanently broken -- it
    does not self-heal) and when a worker times out (the pool is still
    technically usable, but the hung worker occupies a slot forever;
    replacing the pool at least frees up future calls, though the hung
    worker process itself is not killed and its resources leak until it
    naturally exits).
    """
    with _lock:
        # Only replace if another caller hasn't already recovered it.
        if _state["pool"] is stale_pool:
            stale_pool.shutdown(wait=False)
            _state["pool"] = _create_pool()
            logger.warning(
                "MMD process pool replaced with a fresh pool (reason: %s).", reason
            )


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
    """Run func(**kwargs) in the shared MMD process pool.

    A worker crash (from any native bug) leaves the pool permanently broken
    -- ProcessPoolExecutor does not self-heal. If that happens, this call
    still raises BrokenProcessPool (the caller's request failed and its
    result is lost), but the pool is transparently replaced so the *next*
    call can succeed instead of failing forever.

    A worker that hangs instead of crashing (e.g. a native infinite loop)
    would otherwise occupy its slot forever; this call is bounded by
    TRUSTYAI_MMD_TIMEOUT_SECONDS (default 300s) and raises TimeoutError,
    replacing the pool so later calls aren't blocked by the same hang.
    """
    with _lock:
        pool = _state["pool"]
        if pool is None:
            msg = "MMD process pool not started; start_mmd_executor() must run during app startup"
            raise RuntimeError(msg)
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(pool, functools.partial(func, **kwargs)),
            timeout=_timeout_seconds(),
        )
    except BrokenProcessPool:
        _replace_pool(pool, reason="worker crash")
        raise
    except TimeoutError:
        _replace_pool(pool, reason="worker timeout")
        raise
