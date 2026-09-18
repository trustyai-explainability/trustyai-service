"""SQLite-backend-specific tests (in-process, no server required).

Covers behaviors not exercised by the shared parity suite: file-backed
persistence across instances, in-memory concurrency and isolation, the column-type
no-op defaults, the deserialization-error path, and reset semantics.
"""

from __future__ import annotations

import asyncio
import os
import stat
from threading import Event
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy import Connection

pytest.importorskip("sqlalchemy")

from sqlalchemy import event, insert
from sqlalchemy.exc import IntegrityError

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.data.storage.exceptions import DeserializationError
from trustyai_service.service.data.storage.sqlite.sqlite import SQLiteStorage
from trustyai_service.service.health_checks import (
    _health_cache,
    check_storage_readiness,
)


@pytest.mark.asyncio
async def test_memory_instances_are_independent() -> None:
    """Sharing a connection within one storage must not share another's data."""
    first, second = SQLiteStorage(":memory:"), SQLiteStorage(":memory:")
    try:
        await first.write_data("ds", np.array([[1]]), ["x"])
        assert await first.list_all_datasets() == ["ds"]
        assert await second.list_all_datasets() == []
    finally:
        first._engine.dispose()
        second._engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "file"])
async def test_discovery_cannot_rollback_a_write(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery during an insert must not undo a successfully written row."""
    path = ":memory:" if backend == "memory" else str(tmp_path / "concurrent.sqlite")
    storage = SQLiteStorage(path)
    await storage.write_data("model_inputs", np.array([[1]]), ["x"])
    reader_ready, start_read, reader_done = Event(), Event(), Event()
    original_discovery = storage._list_all_datasets_sync

    def concurrent_discovery() -> list[str]:
        reader_ready.set()
        assert start_read.wait(5)
        try:
            return original_discovery()
        finally:
            reader_done.set()

    def pause_after_insert(
        _conn: Connection, _cursor: object, statement: str, *_args: object
    ) -> None:
        if statement.startswith("INSERT INTO trustyai_v2_dataset_"):
            start_read.set()
            # Allow the reader to complete if it can share the writer's
            # connection. Safe serialization instead keeps it waiting until
            # this transaction commits; the writer must not wait indefinitely.
            reader_done.wait(0.5)

    monkeypatch.setattr(storage, "_list_all_datasets_sync", concurrent_discovery)
    event.listen(storage._engine, "after_cursor_execute", pause_after_insert)
    discovery = asyncio.create_task(storage.list_all_datasets())
    try:
        assert await asyncio.to_thread(reader_ready.wait, 5)
        await storage.write_data("model_inputs", np.array([[2]]), ["x"])
        assert await discovery == ["model_inputs"]
        assert await storage.dataset_rows("model_inputs") == 2
        np.testing.assert_array_equal(
            await storage.read_data("model_inputs"), np.array([[1], [2]])
        )
    finally:
        start_read.set()
        try:
            await discovery
        finally:
            event.remove(storage._engine, "after_cursor_execute", pause_after_insert)
            storage._engine.dispose()


@pytest.mark.asyncio
async def test_failed_write_releases_the_memory_connection() -> None:
    """A rolled-back write must leave the shared connection usable by a reader."""
    storage = SQLiteStorage(":memory:")
    await storage.write_data("ds", np.array([[1]]), ["x"])

    def fail_after_insert(
        _conn: Connection, _cursor: object, statement: str, *_args: object
    ) -> None:
        if statement.startswith("INSERT INTO trustyai_v2_dataset_"):
            msg = "simulated write failure"
            raise RuntimeError(msg)

    try:
        event.listen(storage._engine, "after_cursor_execute", fail_after_insert)
        try:
            with pytest.raises(RuntimeError, match="simulated write failure"):
                await storage.write_data("ds", np.array([[2]]), ["x"])
        finally:
            event.remove(storage._engine, "after_cursor_execute", fail_after_insert)
        assert await asyncio.wait_for(storage.list_all_datasets(), timeout=5) == ["ds"]
        await storage.write_data("ds", np.array([[3]]), ["x"])
        assert await storage.dataset_rows("ds") == 2
        np.testing.assert_array_equal(
            await storage.read_data("ds"), np.array([[1], [3]])
        )
    finally:
        storage._engine.dispose()


@pytest.mark.asyncio
async def test_file_backed_persists_across_instances(tmp_path: Path) -> None:
    """Data written by one file-backed instance is visible to a new instance."""
    db_path = str(tmp_path / "trustyai.sqlite")
    writer = SQLiteStorage(db_path)
    data = np.arange(12).reshape(4, 3)
    await writer.write_data("ds", data, ["a", "b", "c"])

    reader = SQLiteStorage(db_path)
    assert await reader.dataset_exists("ds")
    assert np.array_equal(await reader.read_data("ds"), data)


@pytest.mark.asyncio
async def test_duplicate_dataset_name_rejected() -> None:
    """The UNIQUE constraint on dataset_name blocks a concurrent-create race."""
    storage = SQLiteStorage(":memory:")
    with storage._engine.begin() as conn:
        conn.execute(
            insert(storage._ref).values(dataset_name="d", metadata={}, n_rows=0)
        )
    # A second row with the same dataset_name must be refused by the database.
    with pytest.raises(IntegrityError), storage._engine.begin() as conn:
        conn.execute(
            insert(storage._ref).values(dataset_name="d", metadata={}, n_rows=0)
        )


@pytest.mark.asyncio
async def test_reset_database_then_dataset_gone() -> None:
    """After reset_database, previously-existing datasets no longer exist."""
    storage = SQLiteStorage(":memory:")
    await storage.write_data("ds", np.arange(6).reshape(2, 3), ["a", "b", "c"])
    assert await storage.dataset_exists("ds")
    await storage.reset_database()
    # Reference table dropped -> dataset_exists swallows the missing-table error.
    assert await storage.dataset_exists("ds") is False


@pytest.mark.asyncio
async def test_require_existing_dataset_accepts_keyword_name() -> None:
    """@require_existing_dataset works when dataset_name is passed as a keyword."""
    storage = SQLiteStorage(":memory:")
    await storage.write_data("ds", np.arange(6).reshape(2, 3), ["a", "b", "c"])
    # dataset_name supplied as a keyword -> must not raise IndexError.
    rows = await storage.read_data(dataset_name="ds")
    assert rows.shape == (2, 3)
    with pytest.raises(ValueError, match="does not exist"):
        await storage.read_data(dataset_name="missing")


@pytest.mark.asyncio
async def test_column_type_defaults_are_noops() -> None:
    """set/get_column_types keep the interface defaults (no-op / None)."""
    storage = SQLiteStorage(":memory:")
    await storage.set_column_types("ds", ["int", "float"])  # must not raise
    assert await storage.get_column_types("ds") is None


@pytest.mark.asyncio
async def test_deserialization_error_raised_on_corrupt_payload() -> None:
    """A corrupt stored payload raises DeserializationError, not None."""
    storage = SQLiteStorage(":memory:")
    # Insert raw garbage bytes directly into the payload table.
    with storage._engine.begin() as conn:
        conn.execute(
            insert(storage._payloads).values(
                payload_id="bad", is_input=True, payload_data=b"not-gzip-json"
            )
        )
    with pytest.raises(DeserializationError):
        await storage.get_partial_payload("bad", is_input=True, is_modelmesh=True)

    # sanity: a well-formed payload still round-trips
    await storage.persist_partial_payload(
        PartialPayload(data="dGVzdA=="), "good", is_input=True
    )
    got = await storage.get_partial_payload("good", is_input=True, is_modelmesh=True)
    assert got is not None


@pytest.mark.asyncio
async def test_kserve_input_output_payload_targets() -> None:
    """KServe input/output payloads deserialize to the right classes."""
    storage = SQLiteStorage(":memory:")
    req = KServeInferenceRequest(id="r1", inputs=[])
    resp = KServeInferenceResponse(id="r1", model_name="m", outputs=[])
    await storage.persist_partial_payload(req, "r1", is_input=True)
    await storage.persist_partial_payload(resp, "r1", is_input=False)

    got_in = await storage.get_partial_payload("r1", is_input=True, is_modelmesh=False)
    got_out = await storage.get_partial_payload(
        "r1", is_input=False, is_modelmesh=False
    )
    assert isinstance(got_in, KServeInferenceRequest)
    assert isinstance(got_out, KServeInferenceResponse)


class TestSQLiteHealthCheck:
    """Health-check routing + logic for the SQLITE storage format."""

    def test_memory_is_ready(self) -> None:
        """In-memory SQLite is always ready."""
        _health_cache.cache.clear()
        env = {"SERVICE_STORAGE_FORMAT": "SQLITE", "STORAGE_DATABASE_PATH": ":memory:"}
        with patch.dict(os.environ, env, clear=False):
            result = check_storage_readiness()
        assert result.status == "ok"

    def test_file_in_writable_dir_is_ready(self, tmp_path: Path) -> None:
        """A file-backed SQLite path in a writable dir is ready."""
        _health_cache.cache.clear()
        env = {
            "SERVICE_STORAGE_FORMAT": "SQLITE",
            "STORAGE_DATABASE_PATH": str(tmp_path / "db.sqlite"),
        }
        with patch.dict(os.environ, env, clear=False):
            result = check_storage_readiness()
        assert result.status == "ok"

    def test_file_in_missing_dir_errors(self) -> None:
        """A SQLite path under a nonexistent directory reports an error."""
        _health_cache.cache.clear()
        env = {
            "SERVICE_STORAGE_FORMAT": "SQLITE",
            "STORAGE_DATABASE_PATH": "/nonexistent-dir-xyz/db.sqlite",
        }
        with patch.dict(os.environ, env, clear=False):
            result = check_storage_readiness()
        assert result.status == "error"

    def test_writable_file_in_read_only_dir_errors(self, tmp_path: Path) -> None:
        """A writable database in a read-only directory cannot get a WAL/journal."""
        _health_cache.cache.clear()
        db_dir = tmp_path / "ro-dir"
        db_dir.mkdir()
        db_file = db_dir / "db.sqlite"
        db_file.write_bytes(b"")
        db_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # readable + searchable, not writable
        try:
            env = {
                "SERVICE_STORAGE_FORMAT": "SQLITE",
                "STORAGE_DATABASE_PATH": str(db_file),
            }
            with patch.dict(os.environ, env, clear=False):
                result = check_storage_readiness()
            assert result.status == "error"
        finally:
            db_dir.chmod(stat.S_IRWXU)

    def test_directory_path_errors(self, tmp_path: Path) -> None:
        """A path pointing at a directory is not a usable database file."""
        _health_cache.cache.clear()
        env = {
            "SERVICE_STORAGE_FORMAT": "SQLITE",
            "STORAGE_DATABASE_PATH": str(tmp_path),
        }
        with patch.dict(os.environ, env, clear=False):
            result = check_storage_readiness()
        assert result.status == "error"

    def test_existing_non_writable_file_errors(self, tmp_path: Path) -> None:
        """An existing but read-only database file reports an error."""
        _health_cache.cache.clear()
        db_file = tmp_path / "ro.sqlite"
        db_file.write_bytes(b"")
        db_file.chmod(stat.S_IRUSR)  # read-only for owner
        try:
            env = {
                "SERVICE_STORAGE_FORMAT": "SQLITE",
                "STORAGE_DATABASE_PATH": str(db_file),
            }
            with patch.dict(os.environ, env, clear=False):
                result = check_storage_readiness()
            assert result.status == "error"
        finally:
            db_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
