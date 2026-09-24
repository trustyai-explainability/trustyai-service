"""Tests for process-wide model data coordination locks."""

from __future__ import annotations

import asyncio
import threading

from trustyai_service.service.data.storage.model_locks import get_model_lock


def test_model_lock_registry_reuses_one_lock_per_model() -> None:
    """Use one loop-independent lock for all accesses to one model's datasets."""
    assert get_model_lock("model-a") is get_model_lock("model-a")
    assert get_model_lock("model-a") is not get_model_lock("model-b")


def test_model_lock_coordinates_different_event_loops() -> None:
    """Serialize a writer and reader even when each has its own event loop."""
    lock = get_model_lock("cross-loop-model")
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    errors: list[BaseException] = []

    async def first() -> None:
        async with lock:
            entered.set()
            await asyncio.to_thread(release.wait, 1)

    async def second() -> None:
        second_started.set()
        await asyncio.to_thread(entered.wait, 1)
        async with lock:
            second_entered.set()

    def run_first() -> None:
        try:
            asyncio.run(first())
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    def run_second() -> None:
        try:
            asyncio.run(second())
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    first_thread = threading.Thread(target=run_first)
    second_thread = threading.Thread(target=run_second)
    first_thread.start()
    second_thread.start()
    assert entered.wait(timeout=1)
    assert second_started.wait(timeout=1)
    release.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert not errors
    assert second_entered.is_set()
