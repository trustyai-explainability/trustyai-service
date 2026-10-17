"""Loop-independent asynchronous locks used by storage backends."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Deque


# This is only the stale-handoff polling cadence, never an acquire deadline.
_HANDOFF_MONITOR_INTERVAL = 0.01


@dataclass
class _Waiter:
    """One event-loop-local waiter in a thread-safe lock queue."""

    loop: asyncio.AbstractEventLoop
    event: asyncio.Event
    granted: bool = False
    cancelled: bool = False
    handoff_done: threading.Event = field(default_factory=threading.Event)


class ThreadSafeAsyncLock:
    """An async context-manager lock that can be used by multiple event loops.

    Storage methods are called by the service event loop and by local-explainer
    worker loops. ``asyncio.Lock`` binds its waiters to one event loop, so it
    cannot coordinate those callers. This lock protects its small state machine
    with a thread lock and wakes each waiter on that waiter's own loop.
    """

    def __init__(self) -> None:
        """Initialize an unlocked lock with no loop-bound state."""
        self._state_lock = threading.Lock()
        self._locked = False
        self._handoff: _Waiter | None = None
        self._waiters: Deque[_Waiter] = deque()
        self._active_reference: Self | None = None

    def _retain_self_locked(self) -> None:
        """Keep the lock alive while it is acquired, queued, or handing off."""
        self._active_reference = self

    def _release_self_if_idle_locked(self) -> None:
        """Drop the active-operation reference once the lock is completely idle."""
        if not self._locked and self._handoff is None and not self._waiters:
            self._active_reference = None

    def locked(self) -> bool:
        """Return whether the lock is held or has an owner being scheduled."""
        with self._state_lock:
            return self._locked

    async def acquire(self) -> bool:
        """Acquire the lock, removing a waiter safely if its task is cancelled."""
        waiter = _Waiter(asyncio.get_running_loop(), asyncio.Event())
        with self._state_lock:
            self._reclaim_stale_handoff_locked()
            if not self._locked and not self._waiters:
                self._locked = True
                self._retain_self_locked()
                return True
            self._waiters.append(waiter)
            self._retain_self_locked()

        try:
            await waiter.event.wait()
        except asyncio.CancelledError:
            self._cancel_waiter(waiter)
            raise
        self._acknowledge_waiter(waiter)
        return True

    def _cancel_waiter(self, waiter: _Waiter) -> None:
        """Remove a queued waiter or hand off a lock already granted to it."""
        with self._state_lock:
            if waiter.granted:
                waiter.cancelled = True
                if self._handoff is waiter:
                    self._handoff = None
                    waiter.handoff_done.set()
                    self._grant_next_locked()
                return
            try:
                self._waiters.remove(waiter)
            except ValueError:
                waiter.cancelled = True
            self._release_self_if_idle_locked()

    def _acknowledge_waiter(self, waiter: _Waiter) -> None:
        """Complete a handoff after its event-loop task has resumed."""
        with self._state_lock:
            if waiter.cancelled:
                raise asyncio.CancelledError
            if self._handoff is waiter:
                self._handoff = None
                waiter.handoff_done.set()

    @staticmethod
    def _waiter_loop_is_live(waiter: _Waiter) -> bool:
        """Return whether a waiter can still run its wake-up callback."""
        return not waiter.loop.is_closed() and waiter.loop.is_running()

    def _reclaim_stale_handoff_locked(self) -> None:
        """Reclaim a grant whose waiter loop stopped or closed before ack."""
        waiter = self._handoff
        if waiter is None or self._waiter_loop_is_live(waiter):
            return
        self._reclaim_handoff_locked(waiter)

    def _reclaim_handoff_locked(self, waiter: _Waiter) -> None:
        """Reclaim one stale handoff and continue with the FIFO queue."""
        if self._handoff is not waiter:
            return
        waiter.cancelled = True
        self._handoff = None
        waiter.handoff_done.set()
        self._wake_cancelled_waiter_locked(waiter)
        self._grant_next_locked()

    def _monitor_handoff(self, waiter: _Waiter) -> None:
        """Reclaim an in-flight handoff if its loop stops or closes."""
        while not waiter.handoff_done.wait(_HANDOFF_MONITOR_INTERVAL):
            with self._state_lock:
                if self._handoff is not waiter:
                    return
                if self._waiter_loop_is_live(waiter):
                    continue
                self._reclaim_handoff_locked(waiter)
                return

    def _start_handoff_monitor(self, waiter: _Waiter) -> None:
        """Monitor only this handoff until it is acknowledged or reclaimed."""
        threading.Thread(
            target=self._monitor_handoff,
            args=(waiter,),
            daemon=True,
        ).start()

    @staticmethod
    def _wake_cancelled_waiter_locked(waiter: _Waiter) -> None:
        """Wake a reclaimed waiter so it cannot acquire if its loop resumes."""
        with suppress(RuntimeError):
            waiter.loop.call_soon_threadsafe(waiter.event.set)

    def _grant_next_locked(self) -> None:
        """Grant the lock to the next live waiter while state is held."""
        while self._waiters:
            waiter = self._waiters.popleft()
            if waiter.cancelled:
                continue
            waiter.granted = True
            if not self._waiter_loop_is_live(waiter):
                waiter.cancelled = True
                waiter.handoff_done.set()
                self._wake_cancelled_waiter_locked(waiter)
                continue
            self._handoff = waiter
            try:
                waiter.loop.call_soon_threadsafe(waiter.event.set)
            except RuntimeError:
                # The waiter loop may have been closed during cancellation.
                # Treat it as cancelled and continue the handoff.
                waiter.cancelled = True
                self._handoff = None
                waiter.handoff_done.set()
                self._wake_cancelled_waiter_locked(waiter)
                continue
            self._start_handoff_monitor(waiter)
            return
        self._locked = False
        self._release_self_if_idle_locked()

    def release(self) -> None:
        """Release the lock and wake the next waiter, if one exists."""
        with self._state_lock:
            if not self._locked:
                msg = "Lock is not acquired"
                raise RuntimeError(msg)
            self._grant_next_locked()

    async def __aenter__(self) -> Self:
        """Acquire the lock for an async context manager."""
        await self.acquire()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Release the lock when leaving an async context manager."""
        self.release()
