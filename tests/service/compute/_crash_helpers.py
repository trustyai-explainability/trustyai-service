"""Top-level (picklable) helper functions used by mmd_executor crash tests."""

import os
import signal
import time


def die_hard() -> None:
    """Kill the current process immediately, simulating a native crash."""
    os.kill(os.getpid(), signal.SIGSEGV)


def hang_forever() -> None:
    """Simulate a native call that hangs instead of crashing.

    Sleeps far longer than any test timeout, but bounded (not truly infinite)
    so the orphaned worker process doesn't linger indefinitely after the test
    process exits and stops waiting on it.
    """
    time.sleep(10)


def add_one(x: int) -> int:
    """Trivial picklable function used to exercise the pool's happy path."""
    return x + 1
