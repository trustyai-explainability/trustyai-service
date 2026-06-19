"""Tests for timeout handling and Prometheus metrics in PVC migration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import h5py
import numpy as np
import pytest
from prometheus_client import REGISTRY

from src.service.data.storage.maria.pvc_migration import (
    DEFAULT_TIMEOUT_SECONDS,
    PVCToDBMigrator,
)


@pytest.fixture
def mock_maria_storage():
    """Create a mock MariaDBStorage instance."""
    storage = MagicMock()
    storage.connection_manager = MagicMock()
    storage.write_data = AsyncMock()
    return storage


class TestTimeoutHandling:
    """Test timeout protection for database writes."""

    @pytest.mark.asyncio
    async def test_write_with_timeout_success(self, mock_maria_storage):
        """Test successful write within timeout."""
        migrator = PVCToDBMigrator(mock_maria_storage)
        data = np.array([[1, 2, 3]])
        column_names = ["col1", "col2", "col3"]

        # Should complete without timeout
        await migrator._write_data_to_maria_with_timeout(
            "test_dataset", data, column_names, timeout_seconds=5
        )

        mock_maria_storage.write_data.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_with_timeout_exceeded(self, mock_maria_storage):
        """Test that timeout raises TimeoutError."""

        # Mock write_data to sleep longer than timeout
        async def slow_write(*_args, **_kwargs):
            await asyncio.sleep(2)

        mock_maria_storage.write_data = AsyncMock(side_effect=slow_write)

        migrator = PVCToDBMigrator(mock_maria_storage)
        data = np.array([[1, 2, 3]])
        column_names = ["col1", "col2", "col3"]

        # Should raise TimeoutError after 1 second
        with pytest.raises(TimeoutError, match=r"Database write timeout.*exceeded"):
            await migrator._write_data_to_maria_with_timeout(
                "test_dataset", data, column_names, timeout_seconds=1
            )

    @pytest.mark.asyncio
    async def test_write_uses_default_timeout(self, mock_maria_storage):
        """Test that default timeout constant is used when not specified."""
        migrator = PVCToDBMigrator(mock_maria_storage)
        data = np.array([[1, 2, 3]])
        column_names = ["col1", "col2", "col3"]

        # Call without timeout_seconds parameter
        with patch(
            "src.service.data.storage.maria.pvc_migration.asyncio.timeout"
        ) as mock_timeout:
            # Mock the context manager
            mock_timeout.return_value.__aenter__ = AsyncMock()
            mock_timeout.return_value.__aexit__ = AsyncMock()

            await migrator._write_data_to_maria_with_timeout(
                "test_dataset", data, column_names
            )

            # Verify default timeout was used
            mock_timeout.assert_called_once_with(DEFAULT_TIMEOUT_SECONDS)


class TestPrometheusMetrics:
    """Test Prometheus metrics incrementation."""

    @pytest.mark.asyncio
    async def test_metrics_incremented_on_successful_migration(
        self, mock_maria_storage, tmp_path
    ):
        """Test that Prometheus metrics are incremented on successful file migration."""
        # Create a simple HDF5 file
        file_path = tmp_path / "test_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            data = np.array([[1, 2, 3], [4, 5, 6]])
            dataset = h5f.create_dataset("test_dataset", data=data)
            dataset.attrs["column_names"] = ["col1", "col2", "col3"]
            dataset.attrs["is_bytes"] = False

        # Mock connection manager with validation support
        mock_conn_mgr = MagicMock()
        mock_cursor = MagicMock()

        # Mock fetchone to return row count for validation queries
        def mock_fetchone(*args, **kwargs):  # noqa: ARG001
            if mock_cursor.execute.call_args:
                call_args = mock_cursor.execute.call_args[0]
                # IN_PROGRESS migration check query (in _start_migration_tracking)
                if (
                    "SELECT id, total_files FROM trustyai_migration_status"
                    in call_args[0]
                ):
                    return None  # No existing IN_PROGRESS migration
                # File already migrated check query (in _is_file_already_migrated)
                if "SELECT id FROM trustyai_file_migration_status" in call_args[0]:
                    return None  # File not yet migrated
                # Migration completion check query (in migrate())
                if (
                    "SELECT id FROM trustyai_migration_status" in call_args[0]
                    and "status=?" in call_args[0]
                ):
                    return None  # Migration not completed yet
                # Validation query returns row count
                if "trustyai_v2_table_reference" in call_args[0]:
                    return (2,)  # 2 rows written
            return None  # Other queries

        mock_cursor.fetchone = MagicMock(side_effect=mock_fetchone)
        mock_cursor.lastrowid = 1
        conn_context = MagicMock()
        conn_context.__enter__ = MagicMock(return_value=(MagicMock(), mock_cursor))
        conn_context.__exit__ = MagicMock(return_value=None)
        mock_conn_mgr.__enter__ = MagicMock(return_value=conn_context.__enter__())
        mock_conn_mgr.__exit__ = MagicMock(return_value=None)
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(tmp_path))

        # Get initial metric values using public Prometheus API
        # Note: Prometheus automatically adds _total suffix to all Counter metrics
        initial_files_total = (
            REGISTRY.get_sample_value("trustyai_migration_files_total") or 0
        )
        initial_files_success = (
            REGISTRY.get_sample_value("trustyai_migration_files_success_total") or 0
        )
        initial_rows_total = (
            REGISTRY.get_sample_value("trustyai_migration_rows_total") or 0
        )

        # Run migration
        await migrator.migrate()

        # Verify metrics were incremented
        assert (
            REGISTRY.get_sample_value("trustyai_migration_files_total")
            == initial_files_total + 1
        )
        assert (
            REGISTRY.get_sample_value("trustyai_migration_files_success_total")
            == initial_files_success + 1
        )
        assert (
            REGISTRY.get_sample_value("trustyai_migration_rows_total")
            == initial_rows_total + 2
        )  # 2 rows

    @pytest.mark.asyncio
    async def test_metrics_incremented_on_failed_file(
        self, mock_maria_storage, tmp_path
    ):
        """Test that failed file counter is incremented on migration failure."""
        # Create a corrupted HDF5 file (empty file with .hdf5 extension)
        file_path = tmp_path / "corrupted_trustyai.hdf5"
        file_path.write_text("not a valid HDF5 file")

        # Mock connection manager with validation support
        mock_conn_mgr = MagicMock()
        mock_cursor = MagicMock()

        # Mock fetchone to return row count for validation queries
        def mock_fetchone(*args, **kwargs):  # noqa: ARG001
            if mock_cursor.execute.call_args:
                call_args = mock_cursor.execute.call_args[0]
                # IN_PROGRESS migration check query (in _start_migration_tracking)
                if (
                    "SELECT id, total_files FROM trustyai_migration_status"
                    in call_args[0]
                ):
                    return None  # No existing IN_PROGRESS migration
                # File already migrated check query (in _is_file_already_migrated)
                if "SELECT id FROM trustyai_file_migration_status" in call_args[0]:
                    return None  # File not yet migrated
                # Migration completion check query (in migrate())
                if (
                    "SELECT id FROM trustyai_migration_status" in call_args[0]
                    and "status=?" in call_args[0]
                ):
                    return None  # Migration not completed yet
                # Validation query returns row count
                if "trustyai_v2_table_reference" in call_args[0]:
                    return (2,)  # 2 rows written
            return None  # Other queries

        mock_cursor.fetchone = MagicMock(side_effect=mock_fetchone)
        mock_cursor.lastrowid = 1
        conn_context = MagicMock()
        conn_context.__enter__ = MagicMock(return_value=(MagicMock(), mock_cursor))
        conn_context.__exit__ = MagicMock(return_value=None)
        mock_conn_mgr.__enter__ = MagicMock(return_value=conn_context.__enter__())
        mock_conn_mgr.__exit__ = MagicMock(return_value=None)
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(tmp_path))

        # Get initial metric value using public Prometheus API
        # Note: Prometheus automatically adds _total suffix to all Counter metrics
        initial_files_failed = (
            REGISTRY.get_sample_value("trustyai_migration_files_failed_total") or 0
        )

        # Run migration (should handle error gracefully)
        await migrator.migrate()

        # Verify failed counter was incremented
        assert (
            REGISTRY.get_sample_value("trustyai_migration_files_failed_total")
            == initial_files_failed + 1
        )
