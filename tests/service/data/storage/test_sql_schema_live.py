"""Force startup DDL races on the same MariaDB/PostgreSQL servers used in CI.

Set TRUSTYAI_REQUIRE_LIVE_SQL_TESTS=1 to fail if a driver or server is missing.
Local runs without those dependencies skip. Every case owns two unique tables;
it never resets the database or touches the service's existing tables.
"""

from __future__ import annotations

import importlib
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import TYPE_CHECKING, Never
from uuid import uuid4

import pytest

if os.environ.get("TRUSTYAI_REQUIRE_LIVE_SQL_TESTS") != "1":
    pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import DBAPIError

from trustyai_service.service.data.storage.sql import schema
from trustyai_service.service.data.storage.sql.engine import mariadb_url, postgres_url

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Engine, ExceptionContext, MetaData, Table


def _unavailable(reason: str) -> Never:
    if os.environ.get("TRUSTYAI_REQUIRE_LIVE_SQL_TESTS") == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


@pytest.fixture(
    params=[
        pytest.param("postgres", marks=pytest.mark.xdist_group("postgres")),
        pytest.param("mariadb", marks=pytest.mark.xdist_group("mariadb")),
    ]
)
def engines(request: pytest.FixtureRequest) -> Iterator[tuple[Engine, Engine]]:
    """Provide independent engines, requiring live backends when CI requests it."""
    postgres = request.param == "postgres"
    driver = "psycopg" if postgres else "mariadb"
    try:
        importlib.import_module(driver)
    except ImportError as error:
        _unavailable(f"Live {request.param} tests require {driver}: {error}")

    port = 5432 if postgres else 3306
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    except OSError as error:
        _unavailable(f"Live {request.param} server unavailable on port {port}: {error}")

    make_url = postgres_url if postgres else mariadb_url
    url = make_url("trustyai", "trustyai", "127.0.0.1", port, "trustyai-database")
    pair = (
        create_engine(url, connect_args={"connect_timeout": 5}),
        create_engine(url, connect_args={"connect_timeout": 5}),
    )
    try:
        # Bound server-side DDL waits as well as the client-side barrier below.
        for engine in pair:
            with engine.begin() as conn:
                if postgres:
                    conn.exec_driver_sql("SET SESSION lock_timeout = '10s'")
                    conn.exec_driver_sql("SET SESSION statement_timeout = '20s'")
                else:
                    conn.exec_driver_sql("SET SESSION lock_wait_timeout = 10")
        yield pair
    finally:
        for engine in pair:
            engine.dispose()


def _assert_schema_usable(
    engines: tuple[Engine, Engine], reference: Table, payloads: Table
) -> None:
    for index, engine in enumerate(engines):
        with engine.begin() as conn:
            conn.execute(
                reference.insert().values(
                    dataset_name=f"model_{index}", metadata={"columns": ["a"]}, n_rows=1
                )
            )
            conn.execute(
                payloads.insert().values(
                    payload_id=f"request_{index}",
                    is_input=True,
                    payload_data=b"payload",
                )
            )
    for engine in engines:
        with engine.connect() as conn:
            assert conn.execute(
                select(reference.c.dataset_name).order_by(reference.c.table_idx)
            ).scalars().all() == ["model_0", "model_1"]
            assert conn.execute(select(payloads.c.payload_data)).scalars().all() == [
                b"payload",
                b"payload",
            ]


@pytest.mark.parametrize("raced_table", ["reference", "payloads"])
def test_concurrent_schema_initialization(
    engines: tuple[Engine, Engine], raced_table: str
) -> None:
    """Both initializers recover from real duplicate DDL and leave usable tables."""
    prefix = f"test_startup_{uuid4().hex}"
    reference_name, payload_name = f"{prefix}_reference", f"{prefix}_payloads"
    metadata = [schema.make_metadata() for _ in engines]
    for tables in metadata:
        schema.build_reference_table(tables, reference_name)
        schema.build_partial_payload_table(tables, payload_name)
    target = reference_name if raced_table == "reference" else payload_name
    barrier = Barrier(2, timeout=10)
    errors: list[DBAPIError] = []

    def record_error(context: ExceptionContext) -> None:
        # MariaDB's existence check also raises an expected missing-table error.
        if (context.statement or "").lstrip().startswith(
            "CREATE TABLE "
        ) and isinstance(context.sqlalchemy_exception, DBAPIError):
            errors.append(context.sqlalchemy_exception)

    def initialize(engine: Engine, tables: MetaData) -> None:
        synchronized = False

        def wait_before_create(
            _conn: object,
            _cursor: object,
            statement: str,
            _parameters: object,
            _context: object,
            _executemany: object,
        ) -> None:
            nonlocal synchronized
            if not synchronized and statement.lstrip().startswith(
                f"CREATE TABLE {target}"
            ):
                synchronized = True
                # Both existence checks must complete before either CREATE runs.
                barrier.wait()

        event.listen(engine, "before_cursor_execute", wait_before_create)
        event.listen(engine, "handle_error", record_error)
        try:
            schema.create_schema(tables, engine)
            assert synchronized
        finally:
            event.remove(engine, "before_cursor_execute", wait_before_create)
            event.remove(engine, "handle_error", record_error)

    try:
        if raced_table == "payloads":
            # Also exercise recovery when only the second startup table is missing.
            metadata[0].tables[reference_name].create(engines[0])
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(initialize, engine, tables)
                for engine, tables in zip(engines, metadata, strict=True)
            ]
            for future in futures:
                future.result(timeout=30)

        assert errors, "The test must observe a real DDL collision"
        assert all(schema._is_concurrent_table_creation(error) for error in errors)
        _assert_schema_usable(
            engines,
            metadata[0].tables[reference_name],
            metadata[0].tables[payload_name],
        )
    finally:
        metadata[0].drop_all(engines[0], checkfirst=True)
