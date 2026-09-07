"""Unit tests for MariaDB helper utilities and adapter glue (no running database).

Covers the connection manager (used by health checks + migration), the
``get_clean_column_names`` utility, the ``require_existing_dataset`` decorator, the
migration-done callback, and payload delegation.

Storage-behavior coverage (write/read/name-mapping/payloads/etc.) now lives in the
shared, live behavior-parity suite ``test_sql_parity.py``, which runs against
in-memory SQLite on every CI run (plus PostgreSQL/MariaDB when a server is up).
That replaces the previous exact-SQL/connection-manager mock assertions, which
were coupled to the raw-SQL internals removed when the backend moved onto
SQLAlchemy Core.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Coroutine

pytest.importorskip("mariadb")

from trustyai_service.service.data.storage.maria.utils import (
    MariaConnectionManager,
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
# MariaConnectionManager
# ===========================================================================


class TestMariaConnectionManager:
    """Tests for MariaConnectionManager context manager."""

    @patch("trustyai_service.service.data.storage.maria.utils.mariadb.connect")
    def test_connect_without_ssl(self, mock_connect: MagicMock) -> None:
        """Connection without SSL does not include ssl_ca or ssl_verify_cert."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = MariaConnectionManager(
            user="user",
            password="pass",  # noqa: S106  # pragma: allowlist secret
            host="localhost",
            port=3306,
            database="testdb",
            ssl_ca=None,
        )
        with mgr as (conn, _cursor):
            assert conn is mock_conn

        call_kwargs = mock_connect.call_args[1]
        assert "ssl_ca" not in call_kwargs
        assert "ssl_verify_cert" not in call_kwargs

    @patch("trustyai_service.service.data.storage.maria.utils.mariadb.connect")
    def test_connect_with_ssl(self, mock_connect: MagicMock) -> None:
        """Connection with SSL includes ssl_ca and ssl_verify_cert=True."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = MariaConnectionManager(
            user="user",
            password="pass",  # noqa: S106  # pragma: allowlist secret
            host="localhost",
            port=3306,
            database="testdb",
            ssl_ca="/path/to/ca.crt",
        )
        with mgr as (conn, _cursor):
            assert conn is mock_conn

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["ssl_ca"] == "/path/to/ca.crt"
        assert call_kwargs["ssl_verify_cert"] is True

    @patch("trustyai_service.service.data.storage.maria.utils.mariadb.connect")
    def test_connection_closed_on_exit(self, mock_connect: MagicMock) -> None:
        """Connection is closed when exiting the context manager."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = MariaConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=3306,
            database="d",
        )
        with mgr:
            pass

        mock_conn.close.assert_called_once()

    @patch("trustyai_service.service.data.storage.maria.utils.mariadb.connect")
    def test_connection_closed_on_exception(self, mock_connect: MagicMock) -> None:
        """Connection is closed even when an exception occurs inside the context."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = MariaConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=3306,
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


# ===========================================================================
# Deprecation
# ===========================================================================


class TestMariaDeprecation:
    """MariaDBStorage emits a deprecation warning on construction."""

    def test_init_logs_deprecation_warning(self) -> None:
        """Constructing MariaDBStorage logs a DEPRECATED warning (no DB needed)."""
        # Stub the shared base __init__ so no real DB connection is attempted.
        with (
            patch(
                "trustyai_service.service.data.storage.maria.maria.SQLStorage.__init__",
                return_value=None,
            ),
            patch(
                "trustyai_service.service.data.storage.maria.maria.build_engine",
                return_value=MagicMock(),
            ),
            patch(
                "trustyai_service.service.data.storage.maria.maria.MariaConnectionManager",
                return_value=MagicMock(),
            ),
        ):
            from trustyai_service.service.data.storage.maria import (  # noqa: PLC0415
                maria as maria_module,
            )

            with patch.object(maria_module.logger, "warning") as mock_warning:
                maria_module.MariaDBStorage(
                    "u",
                    "p",
                    "h",
                    3306,
                    "d",
                    attempt_migration=False,
                )

        assert mock_warning.called
        assert "DEPRECATED" in mock_warning.call_args[0][0]


# ===========================================================================
# MariaDBStorage with mocked connections
# ===========================================================================


class TestMariaDBStorageMocked:
    """Tests for MariaDBStorage methods using mocked database connections."""

    def _make_storage(self) -> Any:  # noqa: ANN401 -- returns mocked MariaDBStorage
        """Create a MariaDBStorage with mocked __init__ to avoid DB connection."""
        with patch(
            "trustyai_service.service.data.storage.maria.maria.MariaDBStorage.__init__",
            return_value=None,
        ):
            from trustyai_service.service.data.storage.maria.maria import (  # noqa: PLC0415
                MariaDBStorage,
            )

            storage = MariaDBStorage.__new__(MariaDBStorage)
            storage.__init__ = MagicMock()  # type: ignore[method-assign]
            storage.schema_prefix = "trustyai_v2"
            storage.dataset_reference_table = "trustyai_v2_table_reference"
            storage.partial_payload_table = "trustyai_v2_partial_payloads"
            storage.connection_manager = MagicMock()
            return storage

    def test_build_table_name(self) -> None:
        """_build_table_name produces the expected format."""
        storage = self._make_storage()
        assert storage._build_table_name(42) == "trustyai_v2_dataset_42"

    def test_on_migration_done_handles_cancelled(self) -> None:
        """_on_migration_done does nothing for cancelled tasks."""
        from trustyai_service.service.data.storage.maria.maria import (  # noqa: PLC0415
            MariaDBStorage,
        )

        mock_task = MagicMock()
        mock_task.cancelled.return_value = True
        # Should not raise
        MariaDBStorage._on_migration_done(mock_task)
        mock_task.exception.assert_not_called()

    def test_on_migration_done_logs_exception(self) -> None:
        """_on_migration_done logs exceptions from failed migration tasks."""
        from trustyai_service.service.data.storage.maria.maria import (  # noqa: PLC0415
            MariaDBStorage,
        )

        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = RuntimeError("migration failed")
        # Should not raise (it logs the exception)
        MariaDBStorage._on_migration_done(mock_task)

    def test_persist_modelmesh_delegates(self) -> None:
        """persist_modelmesh_payload delegates to persist_partial_payload."""
        storage = self._make_storage()

        async def fake_persist(
            _payload: Any,  # noqa: ANN401
            _payload_id: str,
            *,
            is_input: bool,
        ) -> None:
            pass

        storage.persist_partial_payload = MagicMock(side_effect=fake_persist)

        payload = MagicMock()
        _run(storage.persist_modelmesh_payload(payload, "req-1", is_input=True))
        storage.persist_partial_payload.assert_called_once_with(
            payload, "req-1", is_input=True
        )

    def test_delete_modelmesh_delegates(self) -> None:
        """delete_modelmesh_payload delegates to delete_partial_payload."""
        storage = self._make_storage()

        async def fake_delete(
            _payload_id: str,
            *,
            is_input: bool,
        ) -> None:
            pass

        storage.delete_partial_payload = MagicMock(side_effect=fake_delete)

        _run(storage.delete_modelmesh_payload("req-2", is_input=False))
        storage.delete_partial_payload.assert_called_once_with("req-2", is_input=False)
