"""Unit tests for PVC-to-MariaDB migration module."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import h5py
import numpy as np
import pytest

from trustyai_service.service.data.storage.maria.pvc_migration import (
    DEFAULT_BATCH_SIZE,
    MIGRATION_STATUS_COMPLETE,
    MIGRATION_STATUS_FAILED,
    MIGRATION_STATUS_IN_PROGRESS,
    MIGRATION_TYPE_PVC_TO_DB,
    PVCToDBMigrator,
)


@pytest.fixture
def mock_maria_storage() -> MagicMock:
    """Create a mock MariaDBStorage instance."""
    storage = MagicMock()

    # Track total rows written per dataset for validation
    rows_written: dict[str, int] = {}

    # Mock write_data to track rows
    async def mock_write_data(
        dataset_name: str,
        new_rows: np.ndarray,
        column_names: list[str],  # noqa: ARG001
    ) -> None:
        if dataset_name not in rows_written:
            rows_written[dataset_name] = 0
        rows_written[dataset_name] += len(new_rows)

    storage.write_data = AsyncMock(side_effect=mock_write_data)

    # Setup connection manager with proper context manager protocol
    # that returns (conn, cursor) tuple
    cursor = MagicMock()

    # Mock fetchone to return the tracked row count for validation queries
    def mock_fetchone() -> tuple[int] | None:
        # Extract dataset name from the last execute() call for validation
        if cursor.execute.call_args:
            call_args = cursor.execute.call_args[0]
            # Check if this is a validation query (contains trustyai_v2_table_reference)
            if (
                len(call_args) > 0
                and "trustyai_v2_table_reference" in call_args[0]
                and len(call_args) > 1
                and isinstance(call_args[1], tuple)
            ):
                dataset_name = call_args[1][0]
                return (rows_written.get(dataset_name, 0),)
        # For other queries (migration status, file status), return None
        return None

    cursor.fetchone = MagicMock(side_effect=mock_fetchone)
    cursor.fetchall = MagicMock(return_value=[])
    cursor.lastrowid = 1

    conn_mgr = MagicMock()
    conn_mgr.__enter__ = MagicMock(return_value=(MagicMock(), cursor))
    conn_mgr.__exit__ = MagicMock(return_value=None)

    storage.connection_manager = conn_mgr
    return storage


def mock_connection_manager() -> tuple[MagicMock, MagicMock]:
    """Create a mock MariaConnectionManager with context manager support.

    This is a helper function (not a fixture) that returns a tuple of
    (manager, cursor) for setting up connection manager mocks in tests.
    """
    manager = MagicMock()
    cursor = MagicMock()
    cursor.fetchone = MagicMock(return_value=None)
    cursor.fetchall = MagicMock(return_value=[])
    cursor.lastrowid = 1

    # Setup context manager protocol
    conn_context = MagicMock()
    conn_context.__enter__ = MagicMock(return_value=(MagicMock(), cursor))
    conn_context.__exit__ = MagicMock(return_value=None)
    manager.__enter__ = MagicMock(return_value=conn_context.__enter__())
    manager.__exit__ = MagicMock(return_value=None)

    return manager, cursor


@pytest.fixture
def temp_pvc_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for PVC HDF5 files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_hdf5_file(temp_pvc_dir: Path) -> Path:
    """Create a sample HDF5 file with test data."""
    file_path = temp_pvc_dir / "test_model_inputs_trustyai.hdf5"

    with h5py.File(file_path, "w") as h5f:
        # Create dataset with numeric data
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dataset = h5f.create_dataset("test_model_inputs", data=data)

        # Add attributes
        dataset.attrs["column_names"] = ["feature_1", "feature_2", "feature_3"]
        dataset.attrs["is_bytes"] = False

    return file_path


class TestPVCToDBMigratorInit:
    """Test PVCToDBMigrator initialization."""

    def test_init_with_explicit_pvc_folder(self, mock_maria_storage):
        """Test initialization with explicit PVC folder path."""
        migrator = PVCToDBMigrator(
            maria_storage=mock_maria_storage, pvc_folder="/custom/path"
        )

        assert migrator.maria_storage == mock_maria_storage
        assert migrator.pvc_folder == "/custom/path"
        assert migrator._migration_id is None

    def test_init_with_env_var_pvc_folder(self, mock_maria_storage, monkeypatch):
        """Test initialization using STORAGE_DATA_FOLDER env var."""
        monkeypatch.setenv("STORAGE_DATA_FOLDER", "/env/path")

        migrator = PVCToDBMigrator(maria_storage=mock_maria_storage)

        assert migrator.pvc_folder == "/env/path"

    def test_init_with_default_pvc_folder(self, mock_maria_storage, monkeypatch):
        """Test initialization with default PVC folder."""
        monkeypatch.delenv("STORAGE_DATA_FOLDER", raising=False)

        migrator = PVCToDBMigrator(maria_storage=mock_maria_storage)

        assert migrator.pvc_folder == "/inputs"

    def test_init_with_explicit_batch_size(self, mock_maria_storage):
        """Test initialization with explicit batch size."""
        migrator = PVCToDBMigrator(maria_storage=mock_maria_storage, batch_size=5000)

        assert migrator.batch_size == 5000

    def test_init_with_env_var_batch_size(self, mock_maria_storage, monkeypatch):
        """Test initialization using MIGRATION_BATCH_SIZE env var."""
        monkeypatch.setenv("MIGRATION_BATCH_SIZE", "500")

        migrator = PVCToDBMigrator(maria_storage=mock_maria_storage)

        assert migrator.batch_size == 500

    def test_init_with_default_batch_size(self, mock_maria_storage, monkeypatch):
        """Test initialization with default batch size."""
        monkeypatch.delenv("MIGRATION_BATCH_SIZE", raising=False)

        migrator = PVCToDBMigrator(maria_storage=mock_maria_storage)

        assert migrator.batch_size == DEFAULT_BATCH_SIZE


class TestMigrationTableCreation:
    """Test migration tracking table creation."""

    def test_create_migration_tables(self, mock_maria_storage):
        """Test that migration tables are created with correct schema."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        migrator._create_migration_tables()

        # Verify both CREATE TABLE statements were executed
        assert mock_cursor.execute.call_count == 2

        # Check migration_status table creation
        first_call = mock_cursor.execute.call_args_list[0][0][0]
        assert "trustyai_migration_status" in first_call
        assert "migration_type VARCHAR(50)" in first_call
        assert "status VARCHAR(20)" in first_call

        # Check file_migration_status table creation
        second_call = mock_cursor.execute.call_args_list[1][0][0]
        assert "trustyai_file_migration_status" in second_call
        assert "file_name VARCHAR(512)" in second_call
        assert "FOREIGN KEY (migration_id)" in second_call


class TestMigrationStatusTracking:
    """Test migration status tracking operations."""

    def test_check_migration_not_completed(self, mock_maria_storage):
        """Test checking when migration has not completed."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_cursor.fetchone = MagicMock(return_value=None)
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        result = migrator._check_migration_already_completed()

        assert result is False
        mock_cursor.execute.assert_called_once()

    def test_check_migration_already_completed(self, mock_maria_storage):
        """Test checking when migration has completed."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_cursor.fetchone = MagicMock(return_value=(1, MIGRATION_STATUS_COMPLETE))
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        result = migrator._check_migration_already_completed()

        assert result is True

    def test_start_migration_tracking(self, mock_maria_storage):
        """Test starting migration tracking record."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_cursor.fetchone.return_value = None  # No existing IN_PROGRESS migration
        mock_cursor.lastrowid = 42
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        migration_id = migrator._start_migration_tracking(total_files=10)

        assert migration_id == 42
        assert mock_cursor.execute.call_count == 2  # SELECT + INSERT

        # Verify SELECT statement (first call)
        first_call = mock_cursor.execute.call_args_list[0][0]
        assert "SELECT id, total_files FROM trustyai_migration_status" in first_call[0]

        # Verify INSERT statement (second call)
        second_call = mock_cursor.execute.call_args_list[1][0]
        assert "INSERT INTO trustyai_migration_status" in second_call[0]
        assert second_call[1] == (
            MIGRATION_TYPE_PVC_TO_DB,
            MIGRATION_STATUS_IN_PROGRESS,
            10,
        )

    def test_update_migration_progress(self, mock_maria_storage):
        """Test updating migration progress."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        migrator._migration_id = 42
        migrator._update_migration_progress(files_processed=5)

        # Verify UPDATE statement
        call_args = mock_cursor.execute.call_args[0]
        assert "UPDATE trustyai_migration_status" in call_args[0]
        assert "SET files_processed=?" in call_args[0]
        assert call_args[1] == (5, 42)

    def test_mark_migration_complete(self, mock_maria_storage):
        """Test marking migration as complete."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        migrator._migration_id = 42
        migrator._mark_migration_complete()

        # Verify UPDATE statement
        call_args = mock_cursor.execute.call_args[0]
        assert "UPDATE trustyai_migration_status" in call_args[0]
        assert call_args[1] == (MIGRATION_STATUS_COMPLETE, 42)

    def test_mark_migration_failed(self, mock_maria_storage):
        """Test marking migration as failed."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        migrator._migration_id = 42
        migrator._mark_migration_failed("Test error message")

        # Verify UPDATE statement
        call_args = mock_cursor.execute.call_args[0]
        assert "UPDATE trustyai_migration_status" in call_args[0]
        assert call_args[1] == (MIGRATION_STATUS_FAILED, "Test error message", 42)


class TestFileTracking:
    """Test per-file migration tracking."""

    def test_is_file_not_migrated(self, mock_maria_storage):
        """Test checking when file has not been migrated."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_cursor.fetchone = MagicMock(return_value=None)
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        migrator._migration_id = 42
        result = migrator._is_file_already_migrated("test.hdf5")

        assert result is False

    def test_is_file_already_migrated(self, mock_maria_storage):
        """Test checking when file has been migrated."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_cursor.fetchone = MagicMock(return_value=(1,))
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        migrator._migration_id = 42
        result = migrator._is_file_already_migrated("test.hdf5")

        assert result is True

    def test_mark_file_migrated(self, mock_maria_storage):
        """Test marking a file as migrated."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        migrator._migration_id = 42
        migrator._mark_file_migrated("test.hdf5", "test_dataset", rows_migrated=100)

        # Verify INSERT statement with ON DUPLICATE KEY UPDATE
        call_args = mock_cursor.execute.call_args[0]
        assert "INSERT INTO trustyai_file_migration_status" in call_args[0]
        assert "ON DUPLICATE KEY UPDATE" in call_args[0]
        assert call_args[1] == (42, "test.hdf5", "test_dataset", 100)


class TestHDF5Discovery:
    """Test HDF5 file discovery."""

    def test_discover_hdf5_files_success(
        self, mock_maria_storage, temp_pvc_dir, sample_hdf5_file
    ):
        """Test successful discovery of HDF5 files."""
        # Create additional HDF5 files
        (temp_pvc_dir / "model2_outputs_trustyai.hdf5").touch()
        (temp_pvc_dir / "model3_metadata_trustyai.hdf5").touch()
        (temp_pvc_dir / "not_hdf5.txt").touch()  # Should be ignored

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(temp_pvc_dir))
        hdf5_files = migrator._discover_hdf5_files()

        assert len(hdf5_files) == 3
        assert all(f.suffix == ".hdf5" for f in hdf5_files)

    def test_discover_hdf5_files_folder_not_found(self, mock_maria_storage):
        """Test discovery when PVC folder doesn't exist."""
        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder="/nonexistent/path")

        with pytest.raises(FileNotFoundError, match="PVC folder not found"):
            migrator._discover_hdf5_files()

    def test_discover_hdf5_files_empty_directory(
        self, mock_maria_storage, temp_pvc_dir
    ):
        """Test discovery when no HDF5 files exist."""
        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(temp_pvc_dir))
        hdf5_files = migrator._discover_hdf5_files()

        assert len(hdf5_files) == 0


class TestHDF5Reading:
    """Test reading data from HDF5 files."""

    def test_read_hdf5_dataset_numeric_data(self, mock_maria_storage, sample_hdf5_file):
        """Test reading numeric data from HDF5 dataset."""
        migrator = PVCToDBMigrator(mock_maria_storage)

        data, column_names, is_bytes = migrator._read_hdf5_dataset(
            sample_hdf5_file, "test_model_inputs"
        )

        assert data.shape == (2, 3)
        assert column_names == ["feature_1", "feature_2", "feature_3"]
        assert is_bytes is False or is_bytes == np.False_
        np.testing.assert_array_equal(data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_read_hdf5_dataset_not_found(self, mock_maria_storage, sample_hdf5_file):
        """Test reading nonexistent dataset raises error."""
        migrator = PVCToDBMigrator(mock_maria_storage)

        with pytest.raises(ValueError, match=r"Dataset .* not found"):
            migrator._read_hdf5_dataset(sample_hdf5_file, "nonexistent_dataset")

    def test_read_hdf5_dataset_with_serialized_data(
        self, mock_maria_storage, temp_pvc_dir
    ):
        """Test reading serialized (bytes) data from HDF5 dataset."""
        # Create HDF5 file with serialized data
        file_path = temp_pvc_dir / "serialized_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            # Create void type data (simulating serialized rows)
            void_data = np.array([b"serialized1", b"serialized2"], dtype="V12")
            dataset = h5f.create_dataset("test_dataset", data=void_data)
            dataset.attrs["column_names"] = ["mixed_col"]
            dataset.attrs["is_bytes"] = True

        migrator = PVCToDBMigrator(mock_maria_storage)
        _data, column_names, is_bytes = migrator._read_hdf5_dataset(
            file_path, "test_dataset"
        )

        assert is_bytes is True or is_bytes == np.True_
        assert column_names == ["mixed_col"]


class TestDataWriting:
    """Test writing data to MariaDB."""

    @pytest.mark.asyncio
    async def test_write_data_to_maria(self, mock_maria_storage):
        """Test writing migrated data to MariaDB storage."""
        migrator = PVCToDBMigrator(mock_maria_storage)

        data = np.array([[1, 2, 3], [4, 5, 6]])
        column_names = ["col1", "col2", "col3"]

        await migrator._write_data_to_maria("test_dataset", data, column_names)

        # Verify write_data was called with correct arguments
        mock_maria_storage.write_data.assert_awaited_once()
        call_kwargs = mock_maria_storage.write_data.call_args.kwargs
        assert call_kwargs["dataset_name"] == "test_dataset"
        assert call_kwargs["column_names"] == column_names
        np.testing.assert_array_equal(call_kwargs["new_rows"], [[1, 2, 3], [4, 5, 6]])


class TestSingleFileMigration:
    """Test migrating a single HDF5 file."""

    @pytest.mark.asyncio
    async def test_migrate_single_file_success(
        self, mock_maria_storage, sample_hdf5_file
    ):
        """Test successful migration of a single HDF5 file."""
        migrator = PVCToDBMigrator(mock_maria_storage)

        rows_migrated = await migrator._migrate_single_file(sample_hdf5_file)

        assert rows_migrated == 2  # 2 rows in sample file
        mock_maria_storage.write_data.assert_awaited_once()


class TestFullMigration:
    """Test full migration workflow."""

    @pytest.mark.asyncio
    async def test_migrate_already_completed(self, mock_maria_storage):
        """Test migration skips when already completed."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_cursor.fetchone = MagicMock(return_value=(1, MIGRATION_STATUS_COMPLETE))
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)

        with patch.object(migrator, "_discover_hdf5_files") as mock_discover:
            await migrator.migrate()

            # Should not call discover if already completed
            mock_discover.assert_not_called()

    @pytest.mark.asyncio
    async def test_migrate_no_hdf5_files(self, mock_maria_storage, temp_pvc_dir):
        """Test migration with no HDF5 files."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_cursor.fetchone = MagicMock(return_value=None)
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(temp_pvc_dir))

        await migrator.migrate()

        # Should complete without error and log completion message

    @pytest.mark.asyncio
    async def test_migrate_pvc_folder_not_found(self, mock_maria_storage):
        """Test migration when PVC folder doesn't exist."""
        mock_conn_mgr, _mock_cursor = mock_connection_manager()
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder="/nonexistent")

        # Should not raise, just log warning
        await migrator.migrate()

    @pytest.mark.asyncio
    async def test_migrate_success_with_files(
        self, mock_maria_storage, temp_pvc_dir, sample_hdf5_file
    ):
        """Test successful migration with HDF5 files."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_cursor.fetchone = MagicMock(return_value=None)
        mock_cursor.lastrowid = 1
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(temp_pvc_dir))

        await migrator.migrate()

        # Verify data was written
        assert mock_maria_storage.write_data.await_count > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_discover_hdf5_files_path_is_file(self, mock_maria_storage, temp_pvc_dir):
        """Test error when PVC path points to a file instead of directory."""
        file_path = temp_pvc_dir / "not_a_dir.txt"
        file_path.touch()

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(file_path))

        with pytest.raises(NotADirectoryError, match="not a directory"):
            migrator._discover_hdf5_files()

    @pytest.mark.asyncio
    async def test_migrate_file_with_deserialization_error(
        self, mock_maria_storage, temp_pvc_dir
    ):
        """Test handling of deserialization errors in byte data."""
        file_path = temp_pvc_dir / "corrupted_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            # Create corrupted void data that deserialize_rows will fail on
            bad_data = np.array([b"not-valid-gzip-data-xxxxx"], dtype="V30")
            dataset = h5f.create_dataset("test_dataset", data=bad_data)
            dataset.attrs["column_names"] = ["col1"]
            dataset.attrs["is_bytes"] = True

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(temp_pvc_dir))

        with pytest.raises(ValueError, match="Failed to deserialize"):
            await migrator._migrate_single_file(file_path)

    @pytest.mark.asyncio
    async def test_migrate_with_file_failure(
        self, mock_maria_storage, temp_pvc_dir, sample_hdf5_file
    ):
        """Test migration continues when individual file fails."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()

        # Use semantic mock instead of brittle side_effect list
        def mock_fetchone(*args, **kwargs):  # noqa: ARG001
            if mock_cursor.execute.call_args:
                call_args = mock_cursor.execute.call_args[0]
                # Check for existing IN_PROGRESS migration (resume logic)
                if (
                    "SELECT id, total_files FROM trustyai_migration_status"
                    in call_args[0]
                ):
                    return  # No existing migration to resume
                # Check if migration already completed
                if (
                    "SELECT id FROM trustyai_migration_status" in call_args[0]
                    and "status=?" in call_args[0]
                ):
                    return  # Migration not completed yet
                # Check if file already migrated
                if "SELECT id FROM trustyai_file_migration_status" in call_args[0]:
                    return  # File not yet migrated
            return  # Default for other queries

        mock_cursor.fetchone = MagicMock(side_effect=mock_fetchone)
        mock_cursor.lastrowid = 1
        mock_maria_storage.connection_manager = mock_conn_mgr

        # Make write_data fail
        mock_maria_storage.write_data = AsyncMock(side_effect=Exception("DB error"))

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(temp_pvc_dir))

        # Migration should not raise - continues after file error
        await migrator.migrate()

        # Verify _mark_file_failed was called (via INSERT with error_message)
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        assert any("error_message" in call for call in execute_calls)

    @pytest.mark.asyncio
    async def test_migrate_with_partial_status(self, mock_maria_storage, temp_pvc_dir):
        """Test migration sets PARTIAL status when some files fail."""
        # Create two HDF5 files
        file1 = temp_pvc_dir / "file1_trustyai.hdf5"
        file2 = temp_pvc_dir / "file2_trustyai.hdf5"

        for file_path in [file1, file2]:
            with h5py.File(file_path, "w") as h5f:
                data = np.array([[1, 2], [3, 4]])
                dataset = h5f.create_dataset("test_dataset", data=data)
                dataset.attrs["column_names"] = ["col1", "col2"]
                dataset.attrs["is_bytes"] = False

        mock_conn_mgr, mock_cursor = mock_connection_manager()

        # Mock fetchone with smart pattern for validation queries
        def mock_fetchone(*args, **kwargs):  # noqa: ARG001
            if mock_cursor.execute.call_args:
                call_args = mock_cursor.execute.call_args[0]
                # Validation query returns row count
                if "trustyai_v2_table_reference" in call_args[0]:
                    return (2,)  # 2 rows written
            # For other queries (migration status checks, file status)
            return None

        mock_cursor.fetchone = MagicMock(side_effect=mock_fetchone)
        mock_cursor.lastrowid = 1
        mock_maria_storage.connection_manager = mock_conn_mgr

        # First file succeeds, second fails
        mock_maria_storage.write_data = AsyncMock(
            side_effect=[None, Exception("DB error")]
        )

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(temp_pvc_dir))
        await migrator.migrate()

        # Verify PARTIAL status was set
        execute_calls = [
            (call[0][0], call[0][1] if len(call[0]) > 1 else None)
            for call in mock_cursor.execute.call_args_list
        ]
        partial_update = [
            call
            for call in execute_calls
            if "UPDATE trustyai_migration_status" in call[0]
            and call[1]
            and "PARTIAL" in str(call[1])
        ]
        assert len(partial_update) > 0

    @pytest.mark.asyncio
    async def test_migrate_resume_skips_completed_files(
        self, mock_maria_storage, temp_pvc_dir, sample_hdf5_file
    ):
        """Test migration skips files that were already migrated."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()

        # Use semantic mock instead of brittle side_effect list
        def mock_fetchone(*args, **kwargs):  # noqa: ARG001
            if mock_cursor.execute.call_args:
                call_args = mock_cursor.execute.call_args[0]
                # Check for existing IN_PROGRESS migration (resume logic)
                if (
                    "SELECT id, total_files FROM trustyai_migration_status"
                    in call_args[0]
                ):
                    return None  # No existing migration to resume
                # Check if migration already completed
                if (
                    "SELECT id FROM trustyai_migration_status" in call_args[0]
                    and "status=?" in call_args[0]
                ):
                    return None  # Migration not completed yet
                # Check if file already migrated - return ID to indicate it was migrated
                if "SELECT id FROM trustyai_file_migration_status" in call_args[0]:
                    return (1,)  # File already migrated, skip it
            return None  # Default for other queries

        mock_cursor.fetchone = MagicMock(side_effect=mock_fetchone)
        mock_cursor.lastrowid = 1
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage, pvc_folder=str(temp_pvc_dir))
        await migrator.migrate()

        # Verify write_data was NOT called (file was skipped)
        assert mock_maria_storage.write_data.await_count == 0

    def test_mark_methods_with_no_migration_id(self, mock_maria_storage):
        """Test that mark methods safely return when _migration_id is None."""
        mock_conn_mgr, mock_cursor = mock_connection_manager()
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        # _migration_id is None by default

        # These should all return early without executing SQL
        migrator._update_migration_progress(5)
        migrator._mark_migration_complete()
        migrator._mark_migration_partial(3, 2)
        migrator._mark_migration_failed("error")
        migrator._mark_file_migrated("test.hdf5", "dataset", 10)
        migrator._mark_file_failed("test.hdf5", "error")
        result = migrator._is_file_already_migrated("test.hdf5")

        # Verify no SQL was executed
        assert mock_cursor.execute.call_count == 0
        assert (
            result is False
        )  # _is_file_already_migrated returns False when no migration_id


class TestBatchedProcessing:
    """Test batched data processing for large datasets."""

    @pytest.mark.asyncio
    async def test_migrate_large_dataset_in_batches(
        self, mock_maria_storage, temp_pvc_dir
    ):
        """Test that large datasets are migrated in configurable batches."""
        # Create HDF5 file with 2500 rows (will require 3 batches at batch_size=1000)
        file_path = temp_pvc_dir / "large_dataset_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            # Create large dataset
            rng = np.random.default_rng()
            data = rng.random((2500, 3))
            dataset = h5f.create_dataset("test_dataset", data=data)
            dataset.attrs["column_names"] = ["col1", "col2", "col3"]
            dataset.attrs["is_bytes"] = False

        migrator = PVCToDBMigrator(mock_maria_storage, batch_size=1000)

        await migrator._migrate_single_file(file_path)

        # Verify write_data was called 3 times (2500 rows / 1000 batch_size = 3 batches)
        assert mock_maria_storage.write_data.await_count == 3

        # Verify batch sizes: 1000, 1000, 500
        call_args = mock_maria_storage.write_data.call_args_list
        assert len(call_args[0].kwargs["new_rows"]) == 1000  # First batch
        assert len(call_args[1].kwargs["new_rows"]) == 1000  # Second batch
        assert len(call_args[2].kwargs["new_rows"]) == 500  # Final batch

    @pytest.mark.asyncio
    async def test_migrate_small_dataset_single_batch(
        self, mock_maria_storage, temp_pvc_dir
    ):
        """Test that small datasets are migrated in a single batch."""
        # Create HDF5 file with 500 rows (less than default batch_size=10000)
        file_path = temp_pvc_dir / "small_dataset_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            rng = np.random.default_rng()
            data = rng.random((500, 2))
            dataset = h5f.create_dataset("test_dataset", data=data)
            dataset.attrs["column_names"] = ["col1", "col2"]
            dataset.attrs["is_bytes"] = False

        migrator = PVCToDBMigrator(mock_maria_storage)

        await migrator._migrate_single_file(file_path)

        # Verify write_data was called once (500 < 10000)
        assert mock_maria_storage.write_data.await_count == 1
        assert len(mock_maria_storage.write_data.call_args.kwargs["new_rows"]) == 500

    def test_batch_size_configurable_via_env_var(self, mock_maria_storage, monkeypatch):
        """Test that batch size can be configured via environment variable."""
        monkeypatch.setenv("MIGRATION_BATCH_SIZE", "500")

        migrator = PVCToDBMigrator(mock_maria_storage)

        assert migrator.batch_size == 500

    def test_batch_size_configurable_via_constructor(self, mock_maria_storage):
        """Test that batch size can be configured via constructor parameter."""
        migrator = PVCToDBMigrator(mock_maria_storage, batch_size=2000)

        assert migrator.batch_size == 2000
