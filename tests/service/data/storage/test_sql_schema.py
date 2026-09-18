"""Regression tests for concurrent SQL schema initialization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy import Connection

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import DBAPIError

from trustyai_service.service.data.storage.sql import schema
from trustyai_service.service.data.storage.sql.base import SQLStorage


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raced_table",
    ["trustyai_v2_table_reference", "trustyai_v2_partial_payloads"],
)
async def test_concurrent_table_creation_completes_schema(
    tmp_path: Path, raced_table: str
) -> None:
    """A competing initializer can create either table after the existence check."""
    url = f"sqlite:///{tmp_path / 'startup.sqlite'}"
    engine = create_engine(url)
    competitor = create_engine(url)
    raced = False

    def create_competing_schema(
        _conn: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        nonlocal raced
        if not raced and statement.lstrip().startswith(f"CREATE TABLE {raced_table}"):
            raced = True
            SQLStorage(competitor)

    event.listen(engine, "before_cursor_execute", create_competing_schema)
    try:
        storage = SQLStorage(engine)
        assert raced
        assert set(inspect(engine).get_table_names()) == {
            "trustyai_v2_table_reference",
            "trustyai_v2_partial_payloads",
        }
        data = np.array([[1]])
        await storage.write_data("model_inputs", data, ["a"])
        np.testing.assert_array_equal(await storage.read_data("model_inputs"), data)
    finally:
        engine.dispose()
        competitor.dispose()


class _DriverError(Exception):
    """Driver diagnostics without requiring optional database drivers."""

    def __init__(
        self,
        *,
        sqlstate: str | None = None,
        errno: int | None = None,
        constraint: str | None = None,
    ) -> None:
        super().__init__("database error")
        self.sqlstate = sqlstate
        self.errno = errno
        self.diag = SimpleNamespace(constraint_name=constraint)


@pytest.mark.parametrize(
    "driver_error",
    [
        pytest.param(_DriverError(errno=1050), id="mariadb-duplicate-table"),
        pytest.param(_DriverError(sqlstate="42P07"), id="postgres-duplicate-table"),
        pytest.param(
            _DriverError(sqlstate="23505", constraint="pg_class_relname_nsp_index"),
            id="postgres-duplicate-relation",
        ),
        pytest.param(
            _DriverError(sqlstate="23505", constraint="pg_type_typname_nsp_index"),
            id="postgres-duplicate-type",
        ),
    ],
)
def test_duplicate_ddl_retries_in_a_fresh_transaction(
    driver_error: _DriverError,
) -> None:
    """A failed DDL transaction is rolled back before the schema is checked again."""
    engine = create_engine("sqlite://")
    transactions = []
    rollbacks = []

    def fail_first_create(
        conn: Connection,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        if statement.lstrip().startswith("CREATE TABLE trustyai_v2_table_reference"):
            transactions.append(conn.get_transaction())
            if len(transactions) == 1:
                raise DBAPIError(statement, None, driver_error)

    event.listen(engine, "before_cursor_execute", fail_first_create)
    event.listen(engine, "rollback", rollbacks.append)
    try:
        SQLStorage(engine)
        assert len(rollbacks) == 1
        assert len(transactions) == 2
        assert transactions[0] is not transactions[1]
        assert set(inspect(engine).get_table_names()) == {
            "trustyai_v2_table_reference",
            "trustyai_v2_partial_payloads",
        }
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("statement", "driver_error"),
    [
        ("CREATE TABLE t (id INT)", _DriverError(errno=1045)),
        ("CREATE TABLE t (id INT)", _DriverError(sqlstate="42501")),
        ("CREATE TABLE t (id INT)", _DriverError(sqlstate="42601")),
        (
            "CREATE TABLE t (id INT)",
            _DriverError(sqlstate="23505", constraint="dataset_name_key"),
        ),
        (
            "INSERT INTO t VALUES (1)",
            _DriverError(sqlstate="23505", constraint="pg_class_relname_nsp_index"),
        ),
    ],
)
def test_unrelated_schema_errors_are_not_retried(
    statement: str, driver_error: _DriverError, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authentication, invalid DDL, and data errors propagate unchanged."""
    metadata = schema.make_metadata()
    engine = create_engine("sqlite://")
    error = DBAPIError(statement, None, driver_error)
    attempts = []

    def fail_create(*_args: object, **_kwargs: object) -> None:
        attempts.append(True)
        raise error

    monkeypatch.setattr(metadata, "create_all", fail_create)
    try:
        with pytest.raises(DBAPIError) as caught:
            schema.create_schema(metadata, engine)
        assert caught.value is error
        assert len(attempts) == 1
    finally:
        engine.dispose()


def test_repeated_creation_conflicts_exhaust_the_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent duplicate-object failures cannot turn startup into an endless loop."""
    metadata = schema.make_metadata()
    engine = create_engine("sqlite://")
    error = DBAPIError("CREATE TABLE t (id INT)", None, _DriverError(errno=1050))
    attempts = []

    def fail_create(*_args: object, **_kwargs: object) -> None:
        attempts.append(True)
        raise error

    monkeypatch.setattr(metadata, "create_all", fail_create)
    try:
        with pytest.raises(DBAPIError) as caught:
            schema.create_schema(metadata, engine)
        assert caught.value is error
        assert len(attempts) == 3
    finally:
        engine.dispose()
