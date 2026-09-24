"""Process-wide locks coordinating grouped model dataset access."""

from __future__ import annotations

import threading
import weakref

from .locks import ThreadSafeAsyncLock

_MODEL_LOCKS: weakref.WeakValueDictionary[str, ThreadSafeAsyncLock] = (
    weakref.WeakValueDictionary()
)
_MODEL_LOCKS_LOCK = threading.Lock()


def get_model_lock(model_id: str) -> ThreadSafeAsyncLock:
    """Return one loop-independent lock for all datasets belonging to a model."""
    if not isinstance(model_id, str) or not model_id:
        msg = "model_id must be a non-empty string"
        raise ValueError(msg)
    with _MODEL_LOCKS_LOCK:
        lock = _MODEL_LOCKS.get(model_id)
        if lock is None:
            lock = ThreadSafeAsyncLock()
            _MODEL_LOCKS[model_id] = lock
        return lock
