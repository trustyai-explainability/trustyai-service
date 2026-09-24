"""Focused tests for storage concurrency and coordination."""

from __future__ import annotations

import asyncio
import tempfile
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest

import trustyai_service.service.data.storage as storage_module
from trustyai_service.service.data.storage import GlobalStorageInterface
from trustyai_service.service.data.storage.exceptions import StorageError
from trustyai_service.service.data.storage.locks import ThreadSafeAsyncLock
from trustyai_service.service.data.storage.pvc import (
    BYTES_ATTRIBUTE,
    COLUMN_NAMES_ATTRIBUTE,
    H5PYContext,
    PVCStorage,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine


def _run(coro: Coroutine[Any, Any, Any]) -> Any:  # noqa: ANN401 -- generic test runner
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _start_successor(
    lock: ThreadSafeAsyncLock,
) -> tuple[
    threading.Thread,
    threading.Event,
    threading.Event,
    list[BaseException],
]:
    """Start a queued successor and return its probe state."""
    successor_started = threading.Event()
    successor_entered = threading.Event()
    successor_errors: list[BaseException] = []

    async def successor() -> None:
        successor_started.set()
        async with lock:
            successor_entered.set()

    def run_successor() -> None:
        try:
            asyncio.run(successor())
        except BaseException as error:  # noqa: BLE001
            successor_errors.append(error)

    successor_thread = threading.Thread(target=run_successor, daemon=True)
    successor_thread.start()
    return (
        successor_thread,
        successor_started,
        successor_entered,
        successor_errors,
    )


class _PVCFixture:
    """Shared setup for focused PVC concurrency tests."""

    def setup_method(self) -> None:
        """Create a temporary directory and PVC storage instance."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.storage = PVCStorage(self._tmpdir.name)

    def teardown_method(self) -> None:
        """Clean up the temporary directory."""
        self._tmpdir.cleanup()

    def _write_numeric(
        self,
        name: str = "test_ds",
        data: np.ndarray | None = None,
        cols: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Write a simple numeric dataset and return its data and columns."""
        if data is None:
            data = np.arange(12, dtype=np.float64).reshape(4, 3)
        if cols is None:
            cols = [f"col_{i}" for i in range(data.shape[1])]
        _run(self.storage.write_data(name, data, cols))
        return data, cols


class TestPVCConcurrentWrites(_PVCFixture):
    """Tests for atomic PVC dataset creation and append operations."""

    def _synchronize_concurrent_writers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> list[str]:
        """Gate the actual lock boundary and observe forbidden pre-lock shape probes."""
        writer_count = 3
        lock_barrier = threading.Barrier(writer_count)
        calls_lock = threading.Lock()
        lock_calls = 0
        original_get_lock = self.storage.get_lock

        def synchronized_get_lock(dataset_name: str) -> ThreadSafeAsyncLock:
            nonlocal lock_calls
            with calls_lock:
                lock_calls += 1
                is_initial_call = lock_calls <= writer_count
            if is_initial_call:
                lock_barrier.wait(timeout=2)
            return original_get_lock(dataset_name)

        monkeypatch.setattr(self.storage, "get_lock", synchronized_get_lock)

        dataset_shape_calls: list[str] = []
        original_dataset_shape = self.storage.dataset_shape

        async def observed_dataset_shape(dataset_name: str) -> tuple[int, ...]:
            dataset_shape_calls.append(dataset_name)
            return await original_dataset_shape(dataset_name)

        monkeypatch.setattr(self.storage, "dataset_shape", observed_dataset_shape)
        return dataset_shape_calls

    def test_concurrent_appends_preserve_all_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent appends must not overwrite rows based on stale shape reads."""
        self._write_numeric(
            "concurrent_append",
            data=np.array([[0.0]]),
            cols=["value"],
        )
        dataset_shape_calls = self._synchronize_concurrent_writers(monkeypatch)

        def append(value: float) -> None:
            _run(
                self.storage.write_data(
                    "concurrent_append",
                    np.array([[value]]),
                    ["value"],
                )
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(append, value) for value in (1.0, 2.0, 3.0)]
            for future in futures:
                future.result()

        read = _run(self.storage.read_data("concurrent_append"))
        assert read.shape == (4, 1)
        assert np.array_equal(np.sort(read[:, 0]), np.arange(4, dtype=np.float64))
        assert dataset_shape_calls == []

    def test_concurrent_first_writes_create_one_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent first writes must serialize dataset creation and appends."""
        dataset_shape_calls = self._synchronize_concurrent_writers(monkeypatch)

        def write_first_row(value: float) -> None:
            _run(
                self.storage.write_data(
                    "concurrent_create",
                    np.array([[value]]),
                    ["value"],
                )
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(write_first_row, value) for value in (1.0, 2.0, 3.0)
            ]
            for future in futures:
                future.result()

        read = _run(self.storage.read_data("concurrent_create"))
        assert read.shape == (3, 1)
        assert np.array_equal(np.sort(read[:, 0]), np.arange(1, 4, dtype=np.float64))
        assert dataset_shape_calls == []

    def test_failed_append_rolls_back_resize(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed append must not leave extra rows in the dataset."""
        original = np.array([[1.0]], dtype=np.float64)
        self._write_numeric("rollback", data=original, cols=["value"])
        original_setitem = h5py.Dataset.__setitem__

        def fail_append(dataset: h5py.Dataset, key: object, value: object) -> None:
            if dataset.name.endswith("rollback"):
                failure_message = "append assignment failed"
                raise RuntimeError(failure_message)
            original_setitem(dataset, key, value)

        monkeypatch.setattr(h5py.Dataset, "__setitem__", fail_append)
        with pytest.raises(RuntimeError, match="append assignment failed"):
            _run(
                self.storage.write_data(
                    "rollback",
                    np.array([[2.0]], dtype=np.float64),
                    ["value"],
                )
            )

        assert _run(self.storage.dataset_rows("rollback")) == 1
        assert np.array_equal(_run(self.storage.read_data("rollback")), original)

    def test_oversized_void_append_does_not_resize_legacy_dataset(self) -> None:
        """Rejecting an oversized serialized row must preserve legacy row counts."""
        existing = np.zeros((1, 1), dtype="V4")
        with H5PYContext(self.storage, "legacy_void", "a") as db:
            dataset = db.create_dataset(
                "legacy_void",
                data=existing,
                maxshape=(None, 1),
                chunks=True,
                dtype=existing.dtype,
            )
            dataset.attrs[COLUMN_NAMES_ATTRIBUTE] = ["value"]
            dataset.attrs[BYTES_ATTRIBUTE] = True

        oversized = np.zeros((1, 1), dtype="V8")
        with pytest.raises(ValueError, match="exceeds existing dataset capacity"):
            _run(
                self.storage._write_raw_data(
                    "legacy_void",
                    oversized,
                    ["value"],
                    is_bytes=True,
                )
            )

        assert _run(self.storage.dataset_rows("legacy_void")) == 1
        with H5PYContext(self.storage, "legacy_void", "r") as db:
            assert db["legacy_void"].shape == (1, 1)

    def test_rollback_failure_surfaces_storage_inconsistency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Surface both append and rollback failures to callers."""
        self._write_numeric(
            "rollback_failure",
            data=np.array([[1.0]], dtype=np.float64),
            cols=["value"],
        )
        original_setitem = h5py.Dataset.__setitem__
        original_resize = h5py.Dataset.resize

        def fail_assignment(dataset: h5py.Dataset, key: object, value: object) -> None:
            if dataset.name.endswith("rollback_failure"):
                failure_message = "append assignment failed"
                raise RuntimeError(failure_message)
            original_setitem(dataset, key, value)

        def fail_rollback_resize(dataset: h5py.Dataset, size: int, axis: int) -> None:
            if dataset.name.endswith("rollback_failure") and size == 1:
                failure_message = "rollback resize failed"
                raise RuntimeError(failure_message)
            original_resize(dataset, size, axis=axis)

        monkeypatch.setattr(h5py.Dataset, "__setitem__", fail_assignment)
        monkeypatch.setattr(h5py.Dataset, "resize", fail_rollback_resize)

        with pytest.raises(StorageError, match="Storage is inconsistent") as raised:
            _run(
                self.storage.write_data(
                    "rollback_failure",
                    np.array([[2.0]], dtype=np.float64),
                    ["value"],
                )
            )

        assert isinstance(raised.value.__cause__, BaseExceptionGroup)
        failures = raised.value.__cause__.exceptions
        assert any(str(failure) == "append assignment failed" for failure in failures)
        assert any(str(failure) == "rollback resize failed" for failure in failures)


class TestPVCLockLifecycle(_PVCFixture):
    """Tests for canonical PVC dataset lock lifecycle behavior."""

    def test_repeated_delete_reclaims_dataset_locks(self) -> None:
        """Repeated dataset lifecycles must not grow the lock registry."""
        for index in range(32):
            dataset_name = f"cycle_{index}"
            self._write_numeric(dataset_name)
            _run(self.storage.delete_dataset(dataset_name))

        assert self.storage.locks == {}

    def test_get_lock_reference_survives_delete_before_context_entry(self) -> None:
        """A lock returned before use must not be replaced by deletion cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PVCStorage(tmpdir)
            lock = storage.get_lock("deferred")

            _run(storage.delete_dataset("deferred"))

            assert storage.get_lock("deferred") is lock

    def test_acquired_lock_stays_canonical_until_released(self) -> None:
        """An acquired lock must stay registered even if its caller drops it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PVCStorage(tmpdir)

            async def acquire_and_drop_reference() -> None:
                lock = storage.get_lock("active")
                await lock.acquire()
                lock_reference = weakref.ref(lock)
                del lock

                retained_lock = storage.get_lock("active")
                assert retained_lock is lock_reference()
                retained_lock.release()

            _run(acquire_and_drop_reference())
            assert storage.locks == {}

    def test_concurrent_delete_and_recreate_keep_lock_identity(self) -> None:
        """Delete and recreate operations must share the old lock while queued."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PVCStorage(tmpdir)
            dataset_name = "concurrent"
            _run(
                storage.write_data(
                    dataset_name,
                    np.array([[1.0, 2.0]]),
                    ["first", "second"],
                )
            )
            lock = storage.get_lock(dataset_name)

            async def delete_and_recreate() -> None:
                await lock.acquire()
                delete_started = asyncio.Event()
                recreate_started = asyncio.Event()

                async def delete() -> None:
                    delete_started.set()
                    await storage.delete_dataset(dataset_name)

                async def recreate() -> None:
                    recreate_started.set()
                    await storage.write_data(
                        dataset_name,
                        np.array([[3.0, 4.0]]),
                        ["first", "second"],
                    )

                delete_task = asyncio.create_task(delete())
                await delete_started.wait()
                recreate_task = asyncio.create_task(recreate())
                await recreate_started.wait()

                assert storage.get_lock(dataset_name) is lock

                lock.release()
                await delete_task
                assert storage.get_lock(dataset_name) is lock
                await recreate_task

            _run(delete_and_recreate())
            assert _run(storage.dataset_exists(dataset_name)) is True
            assert _run(storage.dataset_rows(dataset_name)) == 1


class TestThreadSafeLockRecovery(_PVCFixture):
    """Tests for loop-independent lock handoff and recovery."""

    def test_dataset_lock_is_safe_across_event_loops(self) -> None:
        """One dataset lock must coordinate worker loops without binding to one loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PVCStorage(tmpdir)
            lock = storage.get_lock("shared")
            owner_entered = threading.Event()
            release_owner = threading.Event()
            contender_started = threading.Event()
            contender_entered = threading.Event()
            errors: list[BaseException] = []

            async def owner() -> None:
                async with lock:
                    owner_entered.set()
                    await asyncio.to_thread(release_owner.wait)

            async def contender() -> None:
                contender_started.set()
                async with lock:
                    contender_entered.set()

            def run(coroutine: object) -> None:
                try:
                    asyncio.run(coroutine)  # type: ignore[arg-type]
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)

            owner_thread = threading.Thread(target=run, args=(owner(),), daemon=True)
            contender_thread = threading.Thread(
                target=run,
                args=(contender(),),
                daemon=True,
            )
            owner_thread.start()
            assert owner_entered.wait(timeout=1)
            contender_thread.start()
            assert contender_started.wait(timeout=1)

            release_owner.set()
            owner_thread.join(timeout=1)
            contender_thread.join(timeout=1)

            assert not owner_thread.is_alive()
            assert not contender_thread.is_alive()
            assert errors == []
            assert contender_entered.is_set()

    def test_cancelled_cross_loop_waiter_does_not_leak_the_dataset_lock(self) -> None:
        """A cancelled worker must leave the lock available to a later worker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PVCStorage(tmpdir)
            lock = storage.get_lock("shared")
            owner_entered = threading.Event()
            release_owner = threading.Event()
            waiter_queued = threading.Event()
            waiter_cancelled = threading.Event()
            successor_entered = threading.Event()
            errors: list[BaseException] = []

            async def owner() -> None:
                async with lock:
                    owner_entered.set()
                    await asyncio.to_thread(release_owner.wait)

            async def cancelled_waiter() -> None:
                waiter = asyncio.create_task(lock.acquire())
                await asyncio.sleep(0)
                waiter_queued.set()
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
                waiter_cancelled.set()

            async def successor() -> None:
                async with lock:
                    successor_entered.set()

            def run(coroutine: object) -> None:
                try:
                    asyncio.run(coroutine)  # type: ignore[arg-type]
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)

            owner_thread = threading.Thread(target=run, args=(owner(),), daemon=True)
            waiter_thread = threading.Thread(
                target=run,
                args=(cancelled_waiter(),),
                daemon=True,
            )
            successor_thread = threading.Thread(
                target=run,
                args=(successor(),),
                daemon=True,
            )
            owner_thread.start()
            assert owner_entered.wait(timeout=1)
            waiter_thread.start()
            assert waiter_queued.wait(timeout=1)
            assert waiter_cancelled.wait(timeout=1)

            release_owner.set()
            owner_thread.join(timeout=1)
            waiter_thread.join(timeout=1)
            successor_thread.start()
            successor_thread.join(timeout=1)

            assert not owner_thread.is_alive()
            assert not waiter_thread.is_alive()
            assert not successor_thread.is_alive()
            assert errors == []
            assert successor_entered.is_set()

    def test_closed_waiter_loop_does_not_leak_a_handoff(self) -> None:  # noqa: PLR0915
        """A callback accepted before loop closure must not strand the lock."""
        lock = ThreadSafeAsyncLock()
        assert asyncio.run(lock.acquire())

        close_waiter_loop = threading.Event()
        waiter_loop_closed = threading.Event()
        waiter_loop_ready = threading.Event()
        waiter_queued = threading.Event()
        waiter_resumed = threading.Event()
        errors: list[BaseException] = []
        waiter_loop_holder: list[asyncio.AbstractEventLoop] = []

        async def waiter() -> None:
            waiter_queued.set()
            await lock.acquire()
            waiter_resumed.set()

        def run_waiter() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            waiter_loop_holder.append(loop)
            task = loop.create_task(waiter())
            try:
                loop.run_until_complete(asyncio.sleep(0))
                waiter_loop_ready.set()
                close_waiter_loop.wait(timeout=1)
                task.cancel()
                loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
                loop.close()
                waiter_loop_closed.set()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                waiter_loop_ready.set()

        waiter_thread = threading.Thread(target=run_waiter, daemon=True)
        waiter_thread.start()
        assert waiter_loop_ready.wait(timeout=1)
        assert errors == []
        assert waiter_queued.is_set()

        # The loop is open but stopped, so call_soon_threadsafe accepts the
        # callback. Close it before it can run or resume the waiter task.
        lock.release()
        close_waiter_loop.set()
        assert waiter_loop_closed.wait(timeout=1)
        waiter_thread.join(timeout=1)

        assert not waiter_thread.is_alive()
        assert errors == []
        assert waiter_loop_holder[0].is_closed()
        assert not waiter_resumed.is_set()

        async def successor() -> None:
            async with lock:
                successor_entered.set()

        successor_thread = threading.Thread(
            target=lambda: asyncio.run(successor()), daemon=True
        )
        successor_entered = threading.Event()
        successor_thread.start()
        assert successor_entered.wait(timeout=1)
        successor_thread.join(timeout=1)
        assert not successor_thread.is_alive()
        assert not lock.locked()

    def test_open_stopped_waiter_loop_does_not_block_queued_successor(self) -> None:
        """An open but stopped waiter loop must not strand a queued successor."""
        lock = ThreadSafeAsyncLock()
        assert asyncio.run(lock.acquire())

        close_waiter_loop = threading.Event()
        waiter_loop_ready = threading.Event()
        waiter_queued = threading.Event()
        waiter_errors: list[BaseException] = []

        def run_waiter() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(waiter())
            try:
                loop.run_until_complete(asyncio.sleep(0))
                waiter_loop_ready.set()
                close_waiter_loop.wait(timeout=2)
            except BaseException as error:  # noqa: BLE001
                waiter_errors.append(error)
                waiter_loop_ready.set()
            finally:
                task.cancel()
                loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
                loop.close()

        async def waiter() -> None:
            waiter_queued.set()
            await lock.acquire()

        waiter_thread = threading.Thread(target=run_waiter, daemon=True)
        waiter_thread.start()
        assert waiter_loop_ready.wait(timeout=1)
        assert waiter_queued.is_set()

        (
            successor_thread,
            successor_started,
            successor_entered,
            successor_errors,
        ) = _start_successor(lock)
        assert successor_started.wait(timeout=1)

        try:
            lock.release()
            assert successor_entered.wait(timeout=1)
        finally:
            close_waiter_loop.set()
            waiter_thread.join(timeout=1)
            successor_thread.join(timeout=1)

        assert not successor_thread.is_alive()
        assert waiter_errors == []
        assert successor_errors == []
        assert not lock.locked()

    def test_close_after_accepted_wakeup_does_not_strand_queued_successor(  # noqa: PLR0915
        self,
    ) -> None:
        """A waiter loop closing after wakeup acceptance must not strand a successor."""
        lock = ThreadSafeAsyncLock()
        assert asyncio.run(lock.acquire())

        allow_loop_stop = threading.Event()
        waiter_loop_blocked = threading.Event()
        waiter_loop_closed = threading.Event()
        waiter_queued = threading.Event()
        waiter_errors: list[BaseException] = []

        def run_waiter() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(waiter())

            def block_loop() -> None:
                waiter_loop_blocked.set()
                allow_loop_stop.wait(timeout=2)
                loop.stop()

            try:
                loop.run_until_complete(asyncio.sleep(0))
                loop.call_soon(block_loop)
                loop.run_forever()
            except BaseException as error:  # noqa: BLE001
                waiter_errors.append(error)
            finally:
                task.cancel()
                loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
                loop.close()
                waiter_loop_closed.set()

        async def waiter() -> None:
            waiter_queued.set()
            await lock.acquire()

        waiter_thread = threading.Thread(target=run_waiter, daemon=True)
        waiter_thread.start()
        assert waiter_loop_blocked.wait(timeout=1)
        assert waiter_queued.is_set()

        (
            successor_thread,
            successor_started,
            successor_entered,
            successor_errors,
        ) = _start_successor(lock)
        assert successor_started.wait(timeout=1)

        try:
            # The target loop is running but blocked in a callback. The
            # wakeup is accepted, then the loop is stopped and closed before
            # the queued wakeup can run.
            lock.release()
            allow_loop_stop.set()
            assert waiter_loop_closed.wait(timeout=1)
            assert successor_entered.wait(timeout=1)
        finally:
            allow_loop_stop.set()
            waiter_thread.join(timeout=1)
            successor_thread.join(timeout=1)

        assert not successor_thread.is_alive()
        assert waiter_errors == []
        assert successor_errors == []
        assert not lock.locked()

    def test_open_stopped_handoff_recovery_survives_repeated_probes(self) -> None:
        """Repeated stopped-loop handoffs must remain recoverable."""
        lock = ThreadSafeAsyncLock()

        for _ in range(16):
            assert asyncio.run(lock.acquire())
            waiter_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(waiter_loop)
            waiter_started = threading.Event()

            async def waiter(started: threading.Event = waiter_started) -> bool:
                started.set()
                return await lock.acquire()

            waiter_task = waiter_loop.create_task(waiter())
            try:
                waiter_loop.run_until_complete(asyncio.sleep(0))
                assert waiter_started.is_set()
                asyncio.set_event_loop(None)
                lock.release()

                successor_entered = threading.Event()

                async def successor(
                    entered: threading.Event = successor_entered,
                ) -> None:
                    async with lock:
                        entered.set()

                successor_thread = threading.Thread(
                    target=lambda: asyncio.run(successor()), daemon=True
                )
                successor_thread.start()
                assert successor_entered.wait(timeout=1)
                successor_thread.join(timeout=1)
                assert not successor_thread.is_alive()
                asyncio.set_event_loop(waiter_loop)
                waiter_task.cancel()
                waiter_loop.run_until_complete(
                    asyncio.gather(waiter_task, return_exceptions=True)
                )
                assert waiter_task.cancelled()
            finally:
                if not waiter_task.done():
                    waiter_task.cancel()
                    waiter_loop.run_until_complete(
                        asyncio.gather(waiter_task, return_exceptions=True)
                    )
                waiter_loop.close()
                asyncio.set_event_loop(None)

        assert not lock.locked()


def test_first_storage_initialization_is_shared_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent first use creates one storage object and one lock registry."""
    worker_count = 8
    start_barrier = threading.Barrier(worker_count)
    factory_started = threading.Event()
    release_factory = threading.Event()
    calls = 0
    calls_lock = threading.Lock()
    instances: list[object] = []

    def create_storage() -> object:
        nonlocal calls
        with calls_lock:
            calls += 1
            instance = object()
            instances.append(instance)
        factory_started.set()
        if not release_factory.wait(timeout=2):
            raise AssertionError
        return instance

    def get_storage() -> object:
        start_barrier.wait(timeout=2)
        return GlobalStorageInterface.get()

    monkeypatch.setattr(storage_module, "get_storage_interface", create_storage)
    GlobalStorageInterface.reset()
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(get_storage) for _ in range(worker_count)]
            assert factory_started.wait(timeout=2)
            release_factory.set()
            results = [future.result(timeout=3) for future in futures]

        assert calls == 1
        assert len(instances) == 1
        assert all(result is instances[0] for result in results)
    finally:
        release_factory.set()
        GlobalStorageInterface.reset()
