"""Top-level (picklable) helper functions used by mmd_executor crash tests."""

import os
import signal


def die_hard() -> None:
    """Kill the current process immediately, simulating a native crash."""
    os.kill(os.getpid(), signal.SIGSEGV)


def add_one(x: int) -> int:
    """Trivial picklable function used to exercise the pool's happy path."""
    return x + 1
