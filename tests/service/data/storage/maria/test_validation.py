"""Tests for HDF5 validation in PVC-to-MariaDB migration."""

from unittest.mock import AsyncMock, MagicMock

import h5py
import numpy as np
import pytest

from trustyai_service.service.data.storage.maria.pvc_migration import PVCToDBMigrator


@pytest.fixture
def mock_maria_storage():
    """Create a mock MariaDBStorage instance."""
    storage = MagicMock()
    storage.connection_manager = MagicMock()
    storage.write_data = AsyncMock()  # Make write_data async
    return storage


def create_mock_connection_manager(fetchone_return=None):
    """Create a mock MariaConnectionManager with context manager support."""
    manager = MagicMock()
    cursor = MagicMock()
    cursor.fetchone = MagicMock(return_value=fetchone_return)

    # Setup context manager protocol
    conn_context = MagicMock()
    conn_context.__enter__ = MagicMock(return_value=(MagicMock(), cursor))
    conn_context.__exit__ = MagicMock(return_value=None)
    manager.__enter__ = MagicMock(return_value=conn_context.__enter__())
    manager.__exit__ = MagicMock(return_value=None)

    return manager, cursor


class TestHDF5StructureValidation:
    """Test HDF5 file structure validation."""

    def test_validate_valid_hdf5_file(self, mock_maria_storage, tmp_path):
        """Test validation passes for valid HDF5 file."""
        # Create valid HDF5 file
        file_path = tmp_path / "valid_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            data = np.array([[1, 2, 3], [4, 5, 6]])
            dataset = h5f.create_dataset("test_dataset", data=data)
            dataset.attrs["column_names"] = ["col1", "col2", "col3"]
            dataset.attrs["is_bytes"] = False

        migrator = PVCToDBMigrator(mock_maria_storage)
        is_valid, error_msg = migrator._validate_hdf5_structure(file_path)

        assert is_valid is True
        assert error_msg == ""

    def test_validate_empty_hdf5_file(self, mock_maria_storage, tmp_path):
        """Test validation fails for HDF5 file with no datasets."""
        # Create empty HDF5 file
        file_path = tmp_path / "empty_trustyai.hdf5"
        with h5py.File(file_path, "w"):
            pass  # No datasets

        migrator = PVCToDBMigrator(mock_maria_storage)
        is_valid, error_msg = migrator._validate_hdf5_structure(file_path)

        assert is_valid is False
        assert "no datasets" in error_msg.lower()

    def test_validate_missing_column_names_attribute(
        self, mock_maria_storage, tmp_path
    ):
        """Test validation fails when dataset missing column_names attribute."""
        # Create HDF5 file without required attribute
        file_path = tmp_path / "no_columns_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            data = np.array([[1, 2, 3]])
            h5f.create_dataset("test_dataset", data=data)
            # Dataset intentionally missing column_names attribute for test

        migrator = PVCToDBMigrator(mock_maria_storage)
        is_valid, error_msg = migrator._validate_hdf5_structure(file_path)

        assert is_valid is False
        assert "column_names" in error_msg
        assert "test_dataset" in error_msg

    def test_validate_empty_dataset_warns_but_passes(
        self, mock_maria_storage, tmp_path, caplog
    ):
        """Test validation passes but warns for empty dataset."""
        # Create HDF5 file with empty dataset
        file_path = tmp_path / "empty_dataset_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            dataset = h5f.create_dataset("empty_dataset", shape=(0, 3))
            dataset.attrs["column_names"] = ["col1", "col2", "col3"]

        migrator = PVCToDBMigrator(mock_maria_storage)
        is_valid, error_msg = migrator._validate_hdf5_structure(file_path)

        # Should pass validation (empty datasets are allowed)
        assert is_valid is True
        assert error_msg == ""

        # But should log a warning
        assert "empty" in caplog.text.lower()

    def test_validate_corrupted_hdf5_file(self, mock_maria_storage, tmp_path):
        """Test validation fails for corrupted HDF5 file."""
        # Create corrupted file (not valid HDF5)
        file_path = tmp_path / "corrupted_trustyai.hdf5"
        file_path.write_text("This is not a valid HDF5 file")

        migrator = PVCToDBMigrator(mock_maria_storage)
        is_valid, error_msg = migrator._validate_hdf5_structure(file_path)

        assert is_valid is False
        assert "validation error" in error_msg.lower()

    def test_validate_multiple_datasets(self, mock_maria_storage, tmp_path):
        """Test validation checks all datasets in file."""
        # Create HDF5 file with multiple datasets
        file_path = tmp_path / "multi_dataset_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            # First dataset - valid
            data1 = np.array([[1, 2]])
            dataset1 = h5f.create_dataset("dataset1", data=data1)
            dataset1.attrs["column_names"] = ["col1", "col2"]

            # Second dataset - missing column_names
            data2 = np.array([[3, 4]])
            h5f.create_dataset("dataset2", data=data2)

        migrator = PVCToDBMigrator(mock_maria_storage)
        is_valid, error_msg = migrator._validate_hdf5_structure(file_path)

        # Should fail because dataset2 is missing column_names
        assert is_valid is False
        assert "dataset2" in error_msg
        assert "column_names" in error_msg


class TestPostMigrationValidation:
    """Test post-migration row count validation."""

    @pytest.mark.asyncio
    async def test_validate_migration_success(self, mock_maria_storage):
        """Test validation passes when row counts match."""
        mock_conn_mgr, mock_cursor = create_mock_connection_manager(
            fetchone_return=(100,)  # 100 rows in DB
        )
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        result = await migrator._validate_migration("test_dataset", expected_rows=100)

        assert result is True
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_migration_row_count_mismatch(self, mock_maria_storage):
        """Test validation fails when row counts don't match."""
        mock_conn_mgr, _mock_cursor = create_mock_connection_manager(
            fetchone_return=(50,)  # 50 rows in DB
        )
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        result = await migrator._validate_migration(
            "test_dataset",
            expected_rows=100,  # Expected 100
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_migration_dataset_not_found(self, mock_maria_storage):
        """Test validation fails when dataset not found in MariaDB."""
        mock_conn_mgr, _mock_cursor = create_mock_connection_manager(
            fetchone_return=None  # Dataset not found
        )
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        result = await migrator._validate_migration("missing_dataset", expected_rows=10)

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_migration_database_error(self, mock_maria_storage):
        """Test validation handles database errors gracefully."""
        mock_conn_mgr = MagicMock()
        # Simulate database error (e.g., missing table/schema)
        mock_conn_mgr.__enter__ = MagicMock(
            side_effect=Exception("Table 'trustyai_v2_table_reference' doesn't exist")
        )
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)
        result = await migrator._validate_migration("test_dataset", expected_rows=100)

        # Should return False instead of raising exception
        assert result is False


class TestMigrationWithValidation:
    """Test that migration uses validation methods."""

    @pytest.mark.asyncio
    async def test_migrate_single_file_with_invalid_structure(
        self, mock_maria_storage, tmp_path
    ):
        """Test migration fails early on invalid HDF5 structure."""
        # Create HDF5 file missing required attributes
        file_path = tmp_path / "invalid_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            data = np.array([[1, 2, 3]])
            h5f.create_dataset("test_dataset", data=data)
            # Missing column_names attribute

        migrator = PVCToDBMigrator(mock_maria_storage)

        # Should raise ValueError during pre-migration validation
        with pytest.raises(ValueError, match="structure validation failed"):
            await migrator._migrate_single_file(file_path)

        # Database write should never be called
        mock_maria_storage.write_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_migrate_single_file_with_post_validation_failure(
        self, mock_maria_storage, tmp_path
    ):
        """Test migration fails if post-migration validation detects mismatch."""
        # Create valid HDF5 file
        file_path = tmp_path / "test_trustyai.hdf5"
        with h5py.File(file_path, "w") as h5f:
            data = np.array([[1, 2, 3], [4, 5, 6]])
            dataset = h5f.create_dataset("test_dataset", data=data)
            dataset.attrs["column_names"] = ["col1", "col2", "col3"]
            dataset.attrs["is_bytes"] = False

        # Mock connection manager for validation query
        mock_conn_mgr = MagicMock()
        mock_cursor = MagicMock()

        # Simulate row count mismatch (expected 2, but DB says 1)
        mock_cursor.fetchone = MagicMock(return_value=(1,))

        conn_context = MagicMock()
        conn_context.__enter__ = MagicMock(return_value=(MagicMock(), mock_cursor))
        conn_context.__exit__ = MagicMock(return_value=None)
        mock_conn_mgr.__enter__ = MagicMock(return_value=conn_context.__enter__())
        mock_conn_mgr.__exit__ = MagicMock(return_value=None)
        mock_maria_storage.connection_manager = mock_conn_mgr

        migrator = PVCToDBMigrator(mock_maria_storage)

        # Should raise ValueError during post-migration validation
        with pytest.raises(ValueError, match="Post-migration validation failed"):
            await migrator._migrate_single_file(file_path)
