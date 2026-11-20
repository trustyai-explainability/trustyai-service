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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Coroutine

pytest.importorskip("mariadb")

from trustyai_service.service.data.storage.maria.legacy_maria_reader import (
    LegacyMariaDBStorageReader,
)
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
# LegacyMariaDBStorageReader
# ===========================================================================


class TestLegacyMariaDBStorageReader:
    """Tests for legacy MariaDB storage reader configuration."""

    @patch(
        "trustyai_service.service.data.storage.maria.legacy_maria_reader.MariaConnectionManager"
    )
    def test_forwards_tls_ca_to_connection_manager(
        self, mock_connection_manager: MagicMock
    ) -> None:
        """Legacy reader forwards the configured CA to its connection manager."""
        LegacyMariaDBStorageReader(
            user="user",
            password="pass",  # noqa: S106  # pragma: allowlist secret
            host="localhost",
            port=3306,
            database="testdb",
            ssl_ca="/path/to/ca.crt",
        )

        mock_connection_manager.assert_called_once_with(
            "user",
            "pass",  # pragma: allowlist secret
            "localhost",
            3306,
            "testdb",
            ssl_ca="/path/to/ca.crt",
        )


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
            storage.connection_manager.__enter__.return_value = (
                MagicMock(),
                MagicMock(),
            )
            return storage

    @patch(
        "trustyai_service.service.data.storage.maria.maria.LegacyMariaDBStorageReader"
    )
    def test_legacy_reader_reuses_database_tls_ca(
        self, mock_legacy_reader: MagicMock
    ) -> None:
        """Legacy migration connections reuse the MariaDB CA certificate."""
        storage = self._make_storage()
        storage.user = "user"
        storage.password = "pass"  # noqa: S105  # pragma: allowlist secret
        storage.host = "localhost"
        storage.port = 3306
        storage.database = "testdb"
        storage.ssl_ca = "/path/to/ca.crt"
        mock_legacy_reader.return_value.legacy_data_exists.return_value = False

        _run(storage._migrate_from_legacy_db())

        mock_legacy_reader.assert_called_once_with(
            user="user",
            password="pass",  # noqa: S106  # pragma: allowlist secret
            host="localhost",
            port=3306,
            database="testdb",
            ssl_ca="/path/to/ca.crt",
        )

    def _make_legacy_storage(self) -> Any:  # noqa: ANN401 -- returns mocked MariaDBStorage
        """Create a mocked storage with the connection settings the legacy reader needs."""
        storage = self._make_storage()
        storage.user = "user"
        storage.password = "pass"  # noqa: S105  # pragma: allowlist secret
        storage.host = "localhost"
        storage.port = 3306
        storage.database = "testdb"
        storage.ssl_ca = None
        return storage

    @pytest.mark.parametrize("legacy_data_exists", [True, False])
    @patch(
        "trustyai_service.service.data.storage.maria.maria.LegacyMariaDBStorageReader"
    )
    def test_legacy_migration_records_complete_status(
        self, mock_legacy_reader: MagicMock, *, legacy_data_exists: bool
    ) -> None:
        """A finished legacy migration, or nothing to migrate, records COMPLETE."""
        storage = self._make_legacy_storage()
        reader = mock_legacy_reader.return_value
        reader.legacy_data_exists.return_value = legacy_data_exists
        reader.migrate_data = AsyncMock()
        _conn, cursor = storage.connection_manager.__enter__.return_value

        _run(storage._migrate_from_legacy_db())

        assert reader.migrate_data.await_count == int(legacy_data_exists)
        insert_call = cursor.execute.call_args_list[-1]
        assert "INSERT INTO trustyai_migration_status" in insert_call.args[0]
        assert insert_call.args[1] == ("LEGACY_DB", "COMPLETE")

    @pytest.mark.parametrize("failing_step", ["legacy_data_exists", "migrate_data"])
    @patch(
        "trustyai_service.service.data.storage.maria.maria.LegacyMariaDBStorageReader"
    )
    def test_legacy_migration_failure_records_failed_status(
        self, mock_legacy_reader: MagicMock, failing_step: str
    ) -> None:
        """A failing legacy check or migration records FAILED and re-raises."""
        storage = self._make_legacy_storage()
        reader = mock_legacy_reader.return_value
        reader.legacy_data_exists.return_value = True
        reader.migrate_data = AsyncMock()
        if failing_step == "legacy_data_exists":
            reader.legacy_data_exists.side_effect = RuntimeError("boom")
        else:
            reader.migrate_data.side_effect = RuntimeError("boom")
        _conn, cursor = storage.connection_manager.__enter__.return_value

        with pytest.raises(RuntimeError, match="boom"):
            _run(storage._migrate_from_legacy_db())

        insert_call = cursor.execute.call_args_list[-1]
        assert "INSERT INTO trustyai_migration_status" in insert_call.args[0]
        assert insert_call.args[1] == ("LEGACY_DB", "FAILED")

    @pytest.mark.parametrize(
        ("legacy_only", "expect_pvc_migration"), [(True, False), (False, True)]
    )
    def test_run_migration_pvc_path_requires_explicit_opt_in(
        self,
        tmp_path: Any,  # noqa: ANN401 -- pytest tmp_path fixture
        monkeypatch: pytest.MonkeyPatch,
        *,
        legacy_only: bool,
        expect_pvc_migration: bool,
    ) -> None:
        """HDF5 files trigger the PVC migration only when migration was explicitly enabled."""
        (tmp_path / "data.hdf5").write_bytes(b"")
        monkeypatch.setenv("STORAGE_DATA_FOLDER", str(tmp_path))
        storage = self._make_storage()
        storage._legacy_only = legacy_only
        storage._migrate_from_pvc = AsyncMock()
        storage._migrate_from_legacy_db = AsyncMock()

        _run(storage._run_migration())

        assert storage._migrate_from_pvc.await_count == int(expect_pvc_migration)
        assert storage._migrate_from_legacy_db.await_count == int(
            not expect_pvc_migration
        )

    def test_run_migration_defaults_to_the_explicit_path(self) -> None:
        """Storage built without legacy_only keeps the original PVC-first behavior."""
        assert self._make_storage()._legacy_only is False

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
