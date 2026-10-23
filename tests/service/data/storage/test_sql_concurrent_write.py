"""Concurrent first-write recovery using independent engines and real threads."""

from __future__ import annotations

import asyncio
import sqlite3
from functools import partial
from threading import Barrier
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, delete, event, insert, select
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

from tests.service.data.storage.sql_test_resources import SQLTestResources
from tests.service.data.storage.test_sql_parity import (
    _FACTORIES,
    _available_backends,
    _inference_batch,
)
from trustyai_service.service.data.storage.sql import base as sql_base
from trustyai_service.service.data.storage.sql.base import SQLStorage
from trustyai_service.service.data.storage.sqlite.sqlite import SQLiteStorage

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy import Connection, ExceptionContext, Table


@pytest.fixture
def names() -> SQLTestResources:
    """Allocate only resources owned by this test."""
    return SQLTestResources()


@pytest.fixture(params=_available_backends())
def writers(
    request: pytest.FixtureRequest, tmp_path: Path, names: SQLTestResources
) -> Iterator[list[SQLStorage]]:
    """Give each writer a separate engine connected to the same database."""
    stores: list[SQLStorage]
    if request.param == "sqlite":
        stores = [SQLiteStorage(str(tmp_path / "competing.sqlite")) for _ in range(2)]
    else:
        stores = [_FACTORIES[request.param]() for _ in range(2)]
    try:
        yield stores
    finally:
        try:
            asyncio.run(names.cleanup(stores[0]))
        finally:
            for store in stores:
                store._engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("matching_columns", [True, False])
async def test_competing_first_batches(
    writers: list[SQLStorage],
    names: SQLTestResources,
    monkeypatch: pytest.MonkeyPatch,
    *,
    matching_columns: bool,
) -> None:
    """Both valid batches survive a real conflict; an incompatible loser is rejected."""
    dataset_names = [names.dataset(s) for s in ("inputs", "outputs", "metadata")]
    batches = [
        _inference_batch(dataset_names, value, f"inference_{value}") for value in (1, 2)
    ]
    if not matching_columns:
        name, data, _ = batches[1][0]
        batches[1][0] = (name, data, ["different_input"])
    barrier = Barrier(2, timeout=10)
    raced = set()
    conflicts = []
    create_dataset = [writer._create_dataset for writer in writers]

    def create_after_both_absence_checks(
        index: int,
        conn: Connection,
        name: str,
        rows: np.ndarray,
        columns: list[str],
    ) -> Table:
        if name == dataset_names[0] and index not in raced:
            raced.add(index)
            barrier.wait()
        return create_dataset[index](conn, name, rows, columns)

    def capture_conflict(context: ExceptionContext) -> None:
        conflicts.append(context.sqlalchemy_exception)

    for index, writer in enumerate(writers):
        monkeypatch.setattr(
            writer, "_create_dataset", partial(create_after_both_absence_checks, index)
        )
        event.listen(writer._engine, "handle_error", capture_conflict)
    try:
        results = await asyncio.gather(
            *(
                asyncio.to_thread(asyncio.run, writer.write_data_batch(batch))
                for writer, batch in zip(writers, batches, strict=True)
            ),
            return_exceptions=True,
        )
    finally:
        for writer in writers:
            event.remove(writer._engine, "handle_error", capture_conflict)

    assert raced == {0, 1}
    creation_conflicts = [
        error
        for error in conflicts
        if isinstance(error, IntegrityError)
        and writers[0]._is_dataset_creation_conflict(error)
    ]
    retried_ddl_conflicts = [
        error
        for error in conflicts
        if isinstance(error, OperationalError)
        and writers[0]._is_concurrent_ddl_error(error)
    ]
    retried_missing_table_conflicts = [
        error
        for error in conflicts
        if isinstance(error, ProgrammingError)
        and writers[0]._is_missing_dataset_table_error(error)
    ]
    assert creation_conflicts
    assert len(creation_conflicts) + len(retried_ddl_conflicts) + len(
        retried_missing_table_conflicts
    ) == len(conflicts)
    if matching_columns:
        assert results == [None, None]
    else:
        assert sum(result is None for result in results) == 1
        failure = next(result for result in results if result is not None)
        assert isinstance(failure, ValueError)
        assert "Column mismatch" in str(failure)

    inputs, outputs, metadata = [
        await writers[0].read_data(name) for name in dataset_names
    ]
    expected_values = [
        index + 1 for index, result in enumerate(results) if result is None
    ]
    assert sorted(inputs[:, 0].tolist()) == expected_values
    np.testing.assert_array_equal(outputs, inputs * 10)
    assert metadata[:, 0].tolist() == [f"inference_{value}" for value in inputs[:, 0]]
    for name in dataset_names:
        assert await writers[0].dataset_rows(name) == len(expected_values)
        assert (await writers[0].list_all_datasets()).count(name) == 1


@pytest.mark.asyncio
async def test_competing_preparers_finish_a_visible_reference(tmp_path: Path) -> None:
    """Exercise MariaDB's published-reference window using a committed SQLite row."""
    writers = [SQLiteStorage(str(tmp_path / "published.sqlite")) for _ in range(2)]
    data = np.array([[1]])
    columns = ["input"]
    with writers[0]._engine.begin() as conn:
        conn.execute(
            insert(writers[0]._ref).values(
                dataset_name="inputs",
                metadata={
                    "column_names": columns,
                    "aliased_names": columns,
                    "shape": [-1, 1],
                },
                n_rows=0,
            )
        )
    barrier = Barrier(2, timeout=10)
    creations = []

    def race_table_creation(
        _conn: Connection, _cursor: object, statement: str, *_args: object
    ) -> None:
        if statement.lstrip().startswith(
            "CREATE TABLE IF NOT EXISTS trustyai_v2_dataset_"
        ):
            creations.append(statement)
            barrier.wait()

    for writer in writers:
        event.listen(writer._engine, "before_cursor_execute", race_table_creation)
    try:
        # Both preparers observe the same committed reference and missing table.
        # This calls the actual MariaDB preparation helper, with SQLite executing
        # its portable CREATE IF NOT EXISTS statement; no live MariaDB is claimed.
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    writer._prepare_implicit_commit_dataset, "inputs", data, columns
                )
                for writer in writers
            )
        )
        assert len(creations) == 2
        await writers[0].write_data("inputs", data, columns)
        await writers[1].write_data("inputs", data * 2, columns)
        np.testing.assert_array_equal(await writers[0].read_data("inputs"), [[1], [2]])
    finally:
        for writer in writers:
            event.remove(writer._engine, "before_cursor_execute", race_table_creation)
            writer._engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("errno", "should_retry"), [(1684, True), (1213, False)])
async def test_prepare_retries_only_concurrent_ddl_in_a_fresh_transaction(
    monkeypatch: pytest.MonkeyPatch, *, errno: int, should_retry: bool
) -> None:
    """Retry the MariaDB metadata race after rolling back its transaction."""
    storage = SQLiteStorage()
    try:
        data = np.array([[1]])
        columns = ["input"]
        await storage.write_data("inputs", data, columns)
        storage._engine.dialect.name = "mariadb"
        connections = []

        def fail_first_inspection(conn: Connection) -> object:
            connections.append(conn)
            if len(connections) == 1:
                statement = "DESCRIBE trustyai_v2_dataset_0"
                raise OperationalError(
                    statement,
                    {},
                    Exception(errno),
                )
            return sqlalchemy_inspect(conn)

        monkeypatch.setattr(sql_base, "inspect", fail_first_inspection)
        if should_retry:
            storage._prepare_implicit_commit_dataset("inputs", data, columns)
            assert len(connections) == 2
            assert connections[0] is not connections[1]
        else:
            with pytest.raises(OperationalError):
                storage._prepare_implicit_commit_dataset("inputs", data, columns)
            assert len(connections) == 1
    finally:
        storage._engine.dispose()


def _integrity_error(dialect: str, key: str = "dataset_name") -> IntegrityError:
    """Construct driver diagnostics without requiring optional client libraries."""
    reference = "trustyai_v2_table_reference"
    if dialect == "postgresql":
        original = Exception("duplicate key")
        original.sqlstate = "23505"
        original.diag = SimpleNamespace(
            table_name=reference, constraint_name=f"{reference}_{key}_key"
        )
    elif dialect == "sqlite":
        original = sqlite3.IntegrityError(
            f"UNIQUE constraint failed: {reference}.{key}"
        )
        original.sqlite_errorcode = sqlite3.SQLITE_CONSTRAINT_UNIQUE
    elif dialect == "mysql":
        # PyMySQL/mysqlclient put the numeric code in args instead of errno.
        original = Exception(
            1062, f"Duplicate entry 'value' for key '{reference}.{key}'"
        )
    else:
        original = Exception(f"Duplicate entry 'value' for key '{key}'")
        original.errno = 1062
    return IntegrityError(
        "INSERT INTO trustyai_v2_table_reference (dataset_name) VALUES (?)",
        {},
        original,
    )


@pytest.mark.parametrize("dialect", ["postgresql", "mariadb", "mysql", "sqlite"])
@pytest.mark.parametrize("key", ["dataset_name", "table_idx", "other_unique_column"])
def test_only_dataset_name_conflicts_are_retryable(dialect: str, key: str) -> None:
    """A duplicate primary key or another unique column must not be retried."""
    storage = SQLStorage(create_engine("sqlite://"))
    try:
        storage._engine.dialect.name = dialect
        error = _integrity_error(dialect, key)
        assert storage._is_dataset_creation_conflict(error) == (key == "dataset_name")
        error.statement = "INSERT INTO unrelated_table (dataset_name) VALUES (?)"
        assert not storage._is_dataset_creation_conflict(error)
    finally:
        storage._engine.dispose()


@pytest.mark.parametrize("dialect", ["postgresql", "mariadb", "mysql", "sqlite"])
def test_other_error_codes_are_not_retryable(dialect: str) -> None:
    """Matching names never make a non-unique-constraint failure retryable."""
    storage = SQLiteStorage()
    error = _integrity_error(dialect)
    if dialect == "postgresql":
        error.orig.sqlstate = "23503"
    elif dialect == "sqlite":
        error.orig.sqlite_errorcode = sqlite3.SQLITE_CONSTRAINT_NOTNULL
    elif dialect == "mysql":
        error.orig.args = (1213, error.orig.args[1])
    else:
        error.orig.errno = 1213
    try:
        storage._engine.dialect.name = dialect
        assert not storage._is_dataset_creation_conflict(error)
    finally:
        storage._engine.dispose()


@pytest.mark.parametrize(
    ("dialect", "errno", "expected_result"),
    [
        ("mariadb", 1684, "retry"),
        ("mysql", 1684, "retry"),
        ("mariadb", 1213, "raise"),
        ("postgresql", 1684, "raise"),
        ("sqlite", 1684, "raise"),
    ],
)
def test_only_mariadb_concurrent_ddl_errors_are_retryable(
    dialect: str, errno: int, expected_result: str
) -> None:
    """Retry only MariaDB/MySQL's known concurrent table-definition error."""
    storage = SQLStorage(create_engine("sqlite://"))
    try:
        storage._engine.dialect.name = dialect
        error = OperationalError(
            "DESCRIBE trustyai_v2_dataset_0",
            {},
            Exception(errno, "Table definition is being modified"),
        )
        assert storage._is_concurrent_ddl_error(error) is (expected_result == "retry")
    finally:
        storage._engine.dispose()


@pytest.mark.parametrize(
    ("dialect", "statement", "errno", "expected_result"),
    [
        (
            "mariadb",
            "INSERT INTO trustyai_v2_dataset_12 (column_0) VALUES (?)",
            1146,
            "retry",
        ),
        (
            "mysql",
            "INSERT INTO trustyai_v2_dataset_12 (column_0) VALUES (?)",
            1146,
            "retry",
        ),
        (
            "postgresql",
            "INSERT INTO trustyai_v2_dataset_12 (column_0) VALUES (?)",
            1146,
            "raise",
        ),
        (
            "mariadb",
            "INSERT INTO unrelated_table (column_0) VALUES (?)",
            1146,
            "raise",
        ),
        (
            "mariadb",
            "SELECT * FROM trustyai_v2_dataset_12",
            1146,
            "raise",
        ),
        (
            "mariadb",
            "INSERT INTO trustyai_v2_dataset_12 (column_0) VALUES (?)",
            1213,
            "raise",
        ),
    ],
)
def test_only_missing_mariadb_dataset_table_inserts_are_retryable(
    dialect: str,
    statement: str,
    errno: int,
    expected_result: str,
) -> None:
    """Retry only MariaDB/MySQL INSERTs for a missing generated dataset table."""
    storage = SQLStorage(create_engine("sqlite://"))
    original = Exception(errno, "Table does not exist")
    error = ProgrammingError(statement, {}, original)
    try:
        storage._engine.dialect.name = dialect
        assert storage._is_missing_dataset_table_error(error) is (
            expected_result == "retry"
        )
    finally:
        storage._engine.dispose()


@pytest.mark.asyncio
async def test_write_batch_retries_transient_missing_dataset_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient missing-table error retries the full atomic batch."""
    storage = SQLiteStorage()
    batch = _inference_batch(["inputs", "outputs", "metadata"], 1, "inference_1")
    original_write = storage._write_data_batch_once
    storage._engine.dialect.name = "mariadb"
    attempts = 0

    def fail_once(
        datasets: list[tuple[str, np.ndarray, list[str]]],
        parameters: list[list[dict[str, bytes]]],
        expected_tables: dict[str, str],
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            statement = "INSERT INTO trustyai_v2_dataset_0 (column_0) VALUES (?)"
            driver_error = Exception(1146, "Table does not exist")
            raise ProgrammingError(statement, {}, driver_error)
        storage._engine.dialect.name = "sqlite"
        original_write(datasets, parameters, expected_tables)

    monkeypatch.setattr(storage, "_write_data_batch_once", fail_once)
    try:
        await storage.write_data_batch(batch)

        assert attempts == 2
        assert (await storage.read_data("inputs")).tolist() == [[1]]
    finally:
        storage._engine.dispose()


@pytest.mark.asyncio
async def test_preparation_does_not_recreate_a_populated_missing_table() -> None:
    """A lost table is corruption, not a fresh-creation race."""
    storage = SQLiteStorage()
    try:
        await storage.write_data("inputs", np.array([[1], [2]]), ["input"])
        table_name = await storage._get_clean_table_name("inputs")
        with storage._engine.begin() as connection:
            storage._dataset_table(table_name, ["column_0"]).drop(connection)
        with pytest.raises(ValueError, match="registry declares 2 rows"):
            storage._prepare_implicit_commit_dataset(
                "inputs", np.array([[3]]), ["input"]
            )
        assert not sqlalchemy_inspect(storage._engine).has_table(table_name)
        assert await storage.dataset_rows("inputs") == 2
    finally:
        storage._engine.dispose()


@pytest.mark.asyncio
async def test_missing_table_retry_does_not_undo_concurrent_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry must retain the reference identity observed by its first attempt."""
    storage = SQLiteStorage()
    original_write = storage._write_data_batch_once
    attempts = 0

    def deleted_during_first_attempt(
        datasets: list[tuple[str, np.ndarray, list[str]]],
        parameters: list[list[dict[str, bytes]]],
        expected_tables: dict[str, str],
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            name, rows, columns = datasets[0]
            table = storage._prepare_implicit_commit_dataset(name, rows, columns)
            expected_tables[name] = table.name
            with storage._engine.begin() as connection:
                connection.execute(
                    delete(storage._ref).where(storage._ref.c.dataset_name == name)
                )
                table.drop(connection)
            # Keep SQLite's DDL types during setup. Only error classification
            # and the retry need the simulated MariaDB dialect.
            storage._engine.dialect.name = "mariadb"
            statement = f"INSERT INTO {table.name} (column_0) VALUES (?)"  # noqa: S608 -- generated table name
            driver_error = Exception(1146, "Table does not exist")
            raise ProgrammingError(statement, {}, driver_error)
        original_write(datasets, parameters, expected_tables)

    monkeypatch.setattr(storage, "_write_data_batch_once", deleted_during_first_attempt)
    try:
        with pytest.raises(ValueError, match="removed or replaced"):
            await storage.write_data("inputs", np.array([[1]]), ["input"])
        assert attempts == 2
        with storage._engine.connect() as connection:
            assert connection.execute(select(storage._ref)).first() is None
    finally:
        storage._engine.dispose()


@pytest.mark.asyncio
async def test_retry_rolls_back_earlier_dataset_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later reference conflict retries only after all earlier writes roll back."""
    storage = SQLiteStorage()
    batch = _inference_batch(["inputs", "outputs", "metadata"], 1, "inference_1")
    original = storage._create_dataset
    attempts = []
    rollbacks = []
    error = _integrity_error("sqlite")

    def fail_first_output(
        conn: Connection, name: str, rows: np.ndarray, columns: list[str]
    ) -> Table:
        if name == "outputs":
            attempts.append(conn.get_transaction())
            if len(attempts) == 1:
                raise error
        return original(conn, name, rows, columns)

    monkeypatch.setattr(storage, "_create_dataset", fail_first_output)
    event.listen(storage._engine, "rollback", rollbacks.append)
    try:
        await storage.write_data_batch(batch)
        assert len(attempts) == 2
        assert attempts[0] is not attempts[1]
        assert rollbacks  # Includes the failed write transaction and preflight.
        for name, data, _ in batch:
            assert await storage.dataset_rows(name) == 1
            np.testing.assert_array_equal(await storage.read_data(name), data)
    finally:
        storage._engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("unrelated", [True, False])
async def test_batch_conflict_retries_are_bounded(
    monkeypatch: pytest.MonkeyPatch, *, unrelated: bool
) -> None:
    """Unrelated errors propagate immediately; repeated expected conflicts stop."""
    storage = SQLiteStorage()
    error = _integrity_error("sqlite", "table_idx" if unrelated else "dataset_name")
    attempts = []

    def fail_batch(*_args: object) -> None:
        attempts.append(True)
        raise error

    monkeypatch.setattr(storage, "_write_data_batch_once", fail_batch)
    try:
        with pytest.raises(IntegrityError) as caught:
            await storage.write_data("dataset", np.array([[1]]), ["input"])
        assert caught.value is error
        assert len(attempts) == (1 if unrelated else 2)
    finally:
        storage._engine.dispose()
