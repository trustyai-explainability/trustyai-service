"""Unit tests for PostgreSQL helper utilities (no running database required).

Covers the connection manager (used by health checks), the ``get_clean_column_names``
utility, and the ``require_existing_dataset`` decorator.

Storage-behavior coverage (write/read/name-mapping/payloads/etc.) now lives in the
shared, live behavior-parity suite ``test_sql_parity.py``, which runs against
in-memory SQLite on every CI run (plus PostgreSQL/MariaDB when a server is up).
That replaces the previous exact-SQL mock assertions, which were coupled to the
raw-SQL/psycopg internals removed when the backend moved onto SQLAlchemy Core.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Coroutine

pytest.importorskip("psycopg")

from trustyai_service.service.data.storage.postgres.utils import (
    PostgresConnectionManager,
    get_clean_column_names,
    require_existing_dataset,
)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:  # noqa: ANN401 -- generic test runner
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# PostgresConnectionManager
# ===========================================================================


class TestPostgresConnectionManager:
    """Tests for PostgresConnectionManager context manager."""

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connect_without_ssl(self, mock_connect: MagicMock) -> None:
        """Connection without SSL does not include sslmode or sslrootcert."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="user",
            password="pass",  # noqa: S106  # pragma: allowlist secret
            host="localhost",
            port=5432,
            database="testdb",
            ssl_ca=None,
        )
        with mgr as (conn, _cursor):
            assert conn is mock_conn

        call_kwargs = mock_connect.call_args[1]
        assert "sslmode" not in call_kwargs
        assert "sslrootcert" not in call_kwargs

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connect_with_ssl(self, mock_connect: MagicMock) -> None:
        """Connection with SSL includes sslmode=verify-full and sslrootcert."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="user",
            password="pass",  # noqa: S106  # pragma: allowlist secret
            host="localhost",
            port=5432,
            database="testdb",
            ssl_ca="/path/to/ca.crt",
        )
        with mgr as (conn, _cursor):
            assert conn is mock_conn

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["sslmode"] == "verify-full"
        assert call_kwargs["sslrootcert"] == "/path/to/ca.crt"

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connect_with_timeout(self, mock_connect: MagicMock) -> None:
        """connect_timeout is passed through to psycopg.connect when provided."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=5432,
            database="d",
            connect_timeout=2,
        )
        with mgr:
            pass

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["connect_timeout"] == 2  # noqa: PLR2004 -- exact timeout value under test

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connect_without_timeout_omits_kwarg(self, mock_connect: MagicMock) -> None:
        """connect_timeout is omitted from connect kwargs when not provided."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=5432,
            database="d",
        )
        with mgr:
            pass

        call_kwargs = mock_connect.call_args[1]
        assert "connect_timeout" not in call_kwargs

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connection_closed_on_exit(self, mock_connect: MagicMock) -> None:
        """Connection is closed when exiting the context manager."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=5432,
            database="d",
        )
        with mgr:
            pass

        mock_conn.close.assert_called_once()

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connection_closed_on_exception(self, mock_connect: MagicMock) -> None:
        """Connection is closed even when an exception occurs inside the context."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=5432,
            database="d",
        )
        msg = "boom"
        with pytest.raises(RuntimeError, match=msg), mgr:
            raise RuntimeError(msg)

        mock_conn.close.assert_called_once()


# ===========================================================================
# get_clean_column_names
# ===========================================================================


class TestGetCleanColumnNames:
    """Tests for get_clean_column_names utility."""

    def test_generates_safe_names(self) -> None:
        """Generates column_0, column_1, ... regardless of input names."""
        result = get_clean_column_names(["'; DROP TABLE--", "normal", "x"])
        assert result == ["column_0", "column_1", "column_2"]

    def test_empty_list(self) -> None:
        """Returns empty list for empty input."""
        assert get_clean_column_names([]) == []


# ===========================================================================
# require_existing_dataset decorator
# ===========================================================================


class TestRequireExistingDataset:
    """Tests for the require_existing_dataset decorator."""

    def test_raises_for_nonexistent_dataset(self) -> None:
        """Decorated function raises ValueError if dataset does not exist."""

        class _FakeStorage:
            async def dataset_exists(self, _name: str) -> bool:
                return False

        @require_existing_dataset
        async def dummy_func(storage: Any, dataset_name: str) -> str:  # noqa: ANN401, ARG001 -- test mock
            return "should not reach"

        with pytest.raises(ValueError, match="does not exist"):
            _run(dummy_func(_FakeStorage(), "ghost_dataset"))

    def test_passes_for_existing_dataset(self) -> None:
        """Decorated function executes normally if dataset exists."""

        class _FakeStorage:
            async def dataset_exists(self, _name: str) -> bool:
                return True

        @require_existing_dataset
        async def dummy_func(storage: Any, dataset_name: str) -> str:  # noqa: ANN401, ARG001 -- test mock
            return "success"

        result = _run(dummy_func(_FakeStorage(), "real_dataset"))
        assert result == "success"
