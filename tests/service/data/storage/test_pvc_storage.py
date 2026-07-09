"""Tests for PVC (HDF5) storage backend edge cases and uncovered paths.

Covers: dataset existence checks, row/shape queries, name mapping operations,
dataset deletion, serialization edge cases (bytes flag, void types),
get_known_models, get_metadata, and error paths.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest

from src.service.data.exceptions import StorageReadError
from src.service.data.storage.exceptions import DeserializationError
from src.service.data.storage.pvc import (
    BYTES_ATTRIBUTE,
    COLUMN_NAMES_ATTRIBUTE,
    MAX_VOID_TYPE_LENGTH,
    H5PYContext,
    MissingH5PYDataError,
    PVCStorage,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine


def _run(coro: Coroutine[Any, Any, Any]) -> Any:  # noqa: ANN401 -- generic test runner
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _PVCFixture:
    """Shared setup for PVC storage tests."""

    def setup_method(self) -> None:
        """Create a temporary directory and PVC storage instance."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = self._tmpdir.name
        self.storage = PVCStorage(self.tmp_path)

    def teardown_method(self) -> None:
        """Clean up the temporary directory."""
        self._tmpdir.cleanup()

    # --- helpers ---
    def _write_numeric(
        self,
        name: str = "test_ds",
        data: np.ndarray | None = None,
        cols: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Write a simple numeric dataset and return (data, column_names)."""
        if data is None:
            data = np.arange(12, dtype=np.float64).reshape(4, 3)
        if cols is None:
            cols = [f"col_{i}" for i in range(data.shape[1])]
        _run(self.storage.write_data(name, data, cols))
        return data, cols

    def _write_mixed(
        self,
        name: str = "mixed_ds",
    ) -> tuple[list[list[Any]], list[str]]:
        """Write a dataset containing non-numeric (serialized) data."""
        rows: list[list[Any]] = [
            ["hello", 1, {"a": "b"}],
            ["world", 2, {"c": "d"}],
        ]
        cols = ["str_col", "int_col", "dict_col"]
        arr = np.array(rows, dtype=object)
        _run(self.storage.write_data(name, arr, cols))
        return rows, cols


# ===========================================================================
# H5PYContext / MissingH5PYDataError unit tests
# ===========================================================================


class TestH5PYContext(_PVCFixture):
    """Tests for the H5PYContext context manager."""

    def test_read_mode_missing_file_raises(self) -> None:
        """Opening a non-existent file in read mode raises MissingH5PYDataError."""
        ctx = H5PYContext(self.storage, "nonexistent", "r")
        with pytest.raises(MissingH5PYDataError, match="nonexistent"):
            ctx.__enter__()

    def test_write_mode_creates_file(self) -> None:
        """Opening in append mode creates the file if it does not exist."""
        ctx = H5PYContext(self.storage, "new_ds", "a")
        with ctx as db:
            assert isinstance(db, h5py.File)

    def test_missing_error_str(self) -> None:
        """MissingH5PYDataError.__str__ includes dataset name."""
        err = MissingH5PYDataError("my_dataset")
        assert "my_dataset" in str(err)


# ===========================================================================
# Basic CRUD operations
# ===========================================================================


class TestPVCDatasetExists(_PVCFixture):
    """Tests for dataset_exists."""

    def test_exists_false_for_new_storage(self) -> None:
        """dataset_exists returns False when nothing has been written."""
        assert _run(self.storage.dataset_exists("nonexistent")) is False

    def test_exists_true_after_write(self) -> None:
        """dataset_exists returns True after writing data."""
        self._write_numeric("ds_a")
        assert _run(self.storage.dataset_exists("ds_a")) is True

    def test_exists_false_after_delete(self) -> None:
        """dataset_exists returns False after deleting the dataset."""
        self._write_numeric("ds_b")
        _run(self.storage.delete_dataset("ds_b"))
        assert _run(self.storage.dataset_exists("ds_b")) is False


class TestPVCListAllDatasets(_PVCFixture):
    """Tests for list_all_datasets."""

    def test_empty_storage(self) -> None:
        """list_all_datasets returns empty list for fresh storage."""
        result = _run(self.storage.list_all_datasets())
        assert result == []

    def test_lists_written_datasets(self) -> None:
        """list_all_datasets includes datasets that have been written."""
        self._write_numeric("alpha")
        self._write_numeric("beta")
        result = sorted(_run(self.storage.list_all_datasets()))
        assert "alpha" in result
        assert "beta" in result


class TestPVCDatasetRowsAndShape(_PVCFixture):
    """Tests for dataset_rows and dataset_shape."""

    def test_rows_returns_correct_count(self) -> None:
        """dataset_rows returns the number of rows written."""
        data = np.arange(15).reshape(5, 3)
        self._write_numeric("ds_rows", data=data)
        assert _run(self.storage.dataset_rows("ds_rows")) == 5  # noqa: PLR2004 -- literal expected

    def test_shape_returns_correct_tuple(self) -> None:
        """dataset_shape returns the full shape of the dataset."""
        data = np.arange(15).reshape(5, 3)
        self._write_numeric("ds_shape", data=data)
        assert _run(self.storage.dataset_shape("ds_shape")) == (5, 3)

    def test_rows_missing_dataset_raises(self) -> None:
        """dataset_rows raises MissingH5PYDataError for non-existent dataset."""
        with pytest.raises(MissingH5PYDataError):
            _run(self.storage.dataset_rows("ghost"))

    def test_shape_missing_dataset_raises(self) -> None:
        """dataset_shape raises MissingH5PYDataError for non-existent dataset."""
        with pytest.raises(MissingH5PYDataError):
            _run(self.storage.dataset_shape("ghost"))


# ===========================================================================
# Read / Write edge cases
# ===========================================================================


class TestPVCReadWrite(_PVCFixture):
    """Tests for read_data and write_data edge cases."""

    def test_read_full_dataset(self) -> None:
        """read_data without range returns all rows."""
        data, _ = self._write_numeric("full_read")
        read = _run(self.storage.read_data("full_read"))
        assert np.array_equal(read, data)

    def test_read_partial_dataset(self) -> None:
        """read_data with start_row and n_rows returns a slice."""
        data, _ = self._write_numeric("partial_read")
        read = _run(self.storage.read_data("partial_read", start_row=1, n_rows=2))
        assert np.array_equal(read, data[1:3])

    def test_read_start_row_beyond_end_returns_empty(self) -> None:
        """read_data with start_row > dataset size returns an empty array."""
        _data, _ = self._write_numeric("beyond_end")
        read = _run(self.storage.read_data("beyond_end", start_row=999))
        assert len(read) == 0

    def test_append_data_with_matching_shape(self) -> None:
        """write_data appends rows when shapes match."""
        data1, cols = self._write_numeric("append_ds")
        data2 = np.arange(100, 106, dtype=np.float64).reshape(2, 3)
        _run(self.storage.write_data("append_ds", data2, cols))

        combined = _run(self.storage.read_data("append_ds"))
        assert combined.shape[0] == data1.shape[0] + data2.shape[0]
        assert np.array_equal(combined[: data1.shape[0]], data1)
        assert np.array_equal(combined[data1.shape[0] :], data2)

    def test_append_shape_mismatch_raises(self) -> None:
        """write_data raises ValueError when appending rows with different column count."""
        self._write_numeric("shape_mismatch")
        bad_data = np.arange(10, dtype=np.float64).reshape(2, 5)
        with pytest.raises(ValueError, match="Mismatch"):
            _run(self.storage.write_data("shape_mismatch", bad_data, ["a"] * 5))

    def test_write_non_ndarray_list(self) -> None:
        """write_data handles plain Python lists of numeric data."""
        plain_list = [[1.0, 2.0], [3.0, 4.0]]
        _run(self.storage.write_data("list_ds", plain_list, ["x", "y"]))
        read = _run(self.storage.read_data("list_ds"))
        assert read.shape == (2, 2)

    def test_write_and_read_non_numeric_data(self) -> None:
        """write_data serializes non-numeric data; read_data deserializes it."""
        _rows, _cols = self._write_mixed("mixed")
        read = _run(self.storage.read_data("mixed"))
        assert read.shape[0] == 2  # noqa: PLR2004 -- literal expected
        # Check string values roundtrip
        assert read[0][0] == "hello"
        assert read[1][0] == "world"

    def test_bytes_flag_mismatch_raises(self) -> None:
        """Appending numeric data to a bytes-flagged dataset raises ValueError.

        Serialized data is stored as a single-column void array. To trigger the
        bytes-flag check (not the shape mismatch), we must match the column count.
        """
        # Write serialized (bytes-flagged) data first -- stored as shape (N, 1)
        self._write_mixed("bytes_flag_test")

        # Append purely numeric data with shape (1, 1) to match column count
        numeric = np.array([[1.0]])
        with pytest.raises(ValueError, match="previously saved as serialized"):
            _run(self.storage.write_data("bytes_flag_test", numeric, ["col"]))

    def test_bytes_flag_mismatch_numeric_to_serialized_raises(self) -> None:
        """Appending serialized data to a numeric dataset raises ValueError."""
        # Write numeric data first with 1 column
        self._write_numeric(
            "numeric_first",
            data=np.array([[1.0], [2.0]]),
            cols=["a"],
        )

        # Append non-numeric data -- serialized to shape (1, 1) matching columns
        mixed = np.array([["hello"]], dtype=object)
        with pytest.raises(ValueError, match="previously saved as numeric"):
            _run(self.storage.write_data("numeric_first", mixed, ["a"]))


# ===========================================================================
# Delete dataset
# ===========================================================================


class TestPVCDeleteDataset(_PVCFixture):
    """Tests for delete_dataset."""

    def test_delete_existing_dataset(self) -> None:
        """delete_dataset removes the dataset from storage."""
        self._write_numeric("to_delete")
        assert _run(self.storage.dataset_exists("to_delete")) is True
        _run(self.storage.delete_dataset("to_delete"))
        assert _run(self.storage.dataset_exists("to_delete")) is False

    def test_delete_nonexistent_dataset_is_noop(self) -> None:
        """delete_dataset silently does nothing for non-existent datasets."""
        _run(self.storage.delete_dataset("never_existed"))  # should not raise

    def test_delete_then_recreate(self) -> None:
        """A dataset can be recreated after deletion."""
        self._write_numeric("recreate")
        _run(self.storage.delete_dataset("recreate"))
        self._write_numeric("recreate")
        assert _run(self.storage.dataset_exists("recreate")) is True


# ===========================================================================
# Name mapping operations
# ===========================================================================


class TestPVCNameMapping(_PVCFixture):
    """Tests for column name aliasing: apply, get, clear."""

    def test_original_column_names(self) -> None:
        """get_original_column_names returns the names set during write."""
        _, cols = self._write_numeric("names_ds")
        original = _run(self.storage.get_original_column_names("names_ds"))
        assert list(original) == cols

    def test_aliased_names_default_to_original(self) -> None:
        """get_aliased_column_names returns originals when no mapping applied."""
        _, cols = self._write_numeric("alias_default")
        aliased = _run(self.storage.get_aliased_column_names("alias_default"))
        assert list(aliased) == cols

    def test_apply_and_get_name_mapping(self) -> None:
        """apply_name_mapping changes aliased names, preserving originals."""
        _, cols = self._write_numeric("alias_test")
        mapping = {cols[0]: "renamed_0", cols[2]: "renamed_2"}
        _run(self.storage.apply_name_mapping("alias_test", mapping))

        original = _run(self.storage.get_original_column_names("alias_test"))
        aliased = _run(self.storage.get_aliased_column_names("alias_test"))

        assert list(original) == cols
        assert list(aliased) == ["renamed_0", cols[1], "renamed_2"]

    def test_clear_name_mapping(self) -> None:
        """clear_name_mapping removes aliases so aliased names fall back to originals."""
        _, cols = self._write_numeric("clear_alias")
        _run(
            self.storage.apply_name_mapping("clear_alias", {cols[0]: "X"}),
        )
        _run(self.storage.clear_name_mapping("clear_alias"))

        aliased = _run(self.storage.get_aliased_column_names("clear_alias"))
        assert list(aliased) == cols

    def test_clear_name_mapping_no_alias_is_noop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """clear_name_mapping on a dataset with no aliases logs a warning."""
        self._write_numeric("no_alias")
        with caplog.at_level(logging.WARNING):
            _run(self.storage.clear_name_mapping("no_alias"))
        assert "was not found" in caplog.text

    def test_clear_name_mapping_missing_dataset_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """clear_name_mapping on a non-existent dataset logs a warning."""
        # Create the HDF5 file but do not write a dataset with the expected name
        name = "empty_ds"
        filename = self.storage._get_filename(name)
        with h5py.File(filename, "a"):
            pass  # create the file but leave it empty
        with caplog.at_level(logging.WARNING):
            _run(self.storage.clear_name_mapping(name))
        assert "dataset was not found" in caplog.text

    def test_get_original_names_missing_raises(self) -> None:
        """get_original_column_names raises for non-existent dataset."""
        with pytest.raises(MissingH5PYDataError):
            _run(self.storage.get_original_column_names("ghost"))

    def test_get_aliased_names_missing_raises(self) -> None:
        """get_aliased_column_names raises for non-existent dataset."""
        with pytest.raises(MissingH5PYDataError):
            _run(self.storage.get_aliased_column_names("ghost"))


# ===========================================================================
# allocate_valid_dataset_name
# ===========================================================================


class TestAllocateDatasetName:
    """Tests for PVCStorage.allocate_valid_dataset_name."""

    def test_protected_prefix_replaced(self) -> None:
        """Protected prefix is replaced to avoid name collisions."""
        name = "trustyai_internal_my_data"
        result = PVCStorage.allocate_valid_dataset_name(name)
        assert result == "inference_my_data"
        assert not result.startswith("trustyai_internal_")

    def test_normal_name_unchanged(self) -> None:
        """Names without the protected prefix pass through unchanged."""
        assert PVCStorage.allocate_valid_dataset_name("normal") == "normal"


# ===========================================================================
# _get_filename / one_file_per_dataset modes
# ===========================================================================


class TestGetFilename(_PVCFixture):
    """Tests for PVCStorage._get_filename in different modes."""

    def test_one_file_per_dataset_true(self) -> None:
        """With one_file_per_dataset=True, each dataset gets its own file."""
        self.storage.one_file_per_dataset = True
        f = self.storage._get_filename("my_model")
        assert "my_model_" in f
        assert f.endswith(self.storage.data_file)

    def test_one_file_per_dataset_false(self) -> None:
        """With one_file_per_dataset=False, all datasets share one file."""
        self.storage.one_file_per_dataset = False
        f = self.storage._get_filename("any_name")
        assert f == self.storage.data_path


# ===========================================================================
# get_known_models
# ===========================================================================


class TestPVCGetKnownModels(_PVCFixture):
    """Tests for get_known_models."""

    def test_empty_storage_returns_empty_list(self) -> None:
        """get_known_models returns [] when nothing is stored."""
        assert _run(self.storage.get_known_models()) == []

    def test_extracts_model_ids_from_suffixed_datasets(self) -> None:
        """get_known_models extracts model IDs by stripping _inputs/_outputs/_metadata."""
        self._write_numeric("model_a_inputs")
        self._write_numeric("model_a_outputs")
        self._write_numeric("model_b_inputs")
        self._write_numeric("model_b_metadata")

        models = sorted(_run(self.storage.get_known_models()))
        assert models == ["model_a", "model_b"]

    def test_skips_internal_datasets(self) -> None:
        """get_known_models skips datasets with the protected prefix.

        NOTE: allocate_valid_dataset_name rewrites the protected prefix, so
        we manually create a file with the protected prefix to simulate
        a dataset created via internal pathways.
        """
        # Create internal dataset file manually (bypassing allocate_valid_dataset_name)
        internal_name = "trustyai_internal_foo_inputs"
        filename = self.storage._get_filename(internal_name)
        with h5py.File(filename, "a") as db:
            ds = db.create_dataset(
                internal_name,
                data=np.array([[1.0]]),
                maxshape=[None, 1],
                chunks=True,
            )
            ds.attrs[COLUMN_NAMES_ATTRIBUTE] = ["col"]
            ds.attrs[BYTES_ATTRIBUTE] = False

        self._write_numeric("regular_inputs")
        models = _run(self.storage.get_known_models())
        assert models == ["regular"]

    def test_ignores_datasets_without_known_suffix(self) -> None:
        """Datasets without _inputs/_outputs/_metadata suffix are ignored."""
        self._write_numeric("random_table")
        models = _run(self.storage.get_known_models())
        assert models == []


# ===========================================================================
# get_metadata
# ===========================================================================


class TestPVCGetMetadata(_PVCFixture):
    """Tests for get_metadata.

    NOTE: The current PVC get_metadata implementation has a bug where it passes
    keyword arguments to ``StorageMetadata()`` instead of a ``StorageMetadataConfig``
    object. This causes a ``TypeError`` that is caught and re-raised as
    ``StorageReadError``. The tests below document this current behaviour and also
    exercise the intermediate metadata-gathering paths (dataset existence checks,
    shape retrieval, column name retrieval) that execute *before* the final
    ``StorageMetadata(...)`` call fails.
    """

    def test_metadata_raises_due_to_constructor_mismatch(self) -> None:
        """get_metadata raises StorageReadError due to StorageMetadata constructor bug.

        The PVC get_metadata passes keyword args instead of StorageMetadataConfig.
        This exercises: dataset_exists, dataset_shape, get_original_column_names,
        get_aliased_column_names, and the error-handling path in get_metadata.
        """
        input_data = np.arange(6, dtype=np.float64).reshape(2, 3)
        output_data = np.arange(4, dtype=np.float64).reshape(2, 2)
        _run(
            self.storage.write_data(
                "mymodel_inputs", input_data, ["in_a", "in_b", "in_c"]
            ),
        )
        _run(
            self.storage.write_data("mymodel_outputs", output_data, ["out_x", "out_y"]),
        )

        with pytest.raises(StorageReadError, match="Failed to retrieve metadata"):
            _run(self.storage.get_metadata("mymodel"))

    def test_metadata_no_data_also_raises(self) -> None:
        """get_metadata for a model with no datasets also raises StorageReadError."""
        with pytest.raises(StorageReadError, match="Failed to retrieve metadata"):
            _run(self.storage.get_metadata("no_such_model"))

    def test_metadata_input_only_raises(self) -> None:
        """get_metadata with only input data exercises shape/name retrieval paths."""
        input_data = np.arange(6, dtype=np.float64).reshape(3, 2)
        _run(
            self.storage.write_data("partial_inputs", input_data, ["a", "b"]),
        )
        with pytest.raises(StorageReadError, match="Failed to retrieve metadata"):
            _run(self.storage.get_metadata("partial"))

    def test_metadata_with_aliased_names_exercises_alias_path(self) -> None:
        """get_metadata exercises the aliased name mapping code path."""
        input_data = np.arange(4, dtype=np.float64).reshape(2, 2)
        _run(
            self.storage.write_data("aliased_inputs", input_data, ["orig1", "orig2"]),
        )
        _run(
            self.storage.apply_name_mapping("aliased_inputs", {"orig1": "alias1"}),
        )

        with pytest.raises(StorageReadError, match="Failed to retrieve metadata"):
            _run(self.storage.get_metadata("aliased"))

    def test_metadata_with_metadata_dataset_exercises_path(self) -> None:
        """get_metadata exercises the metadata_dataset code path."""
        meta_data = np.arange(4, dtype=np.float64).reshape(2, 2)
        _run(
            self.storage.write_data(
                "metamodel_metadata", meta_data, ["meta_a", "meta_b"]
            ),
        )
        with pytest.raises(StorageReadError, match="Failed to retrieve metadata"):
            _run(self.storage.get_metadata("metamodel"))


# ===========================================================================
# Void type edge cases
# ===========================================================================


class TestPVCVoidTypeEdgeCases(_PVCFixture):
    """Tests for void type handling during append operations."""

    def test_void_type_append_larger_than_existing_raises(self) -> None:
        """Appending serialized data with larger void type than existing dataset raises."""
        name = "void_legacy"
        # Manually create a dataset with a small void type to simulate a legacy dataset
        allocated = PVCStorage.allocate_valid_dataset_name(name)
        filename = self.storage._get_filename(allocated)
        small_void = np.array([np.void(b"\x00" * 10)]).reshape(1, 1)
        with h5py.File(filename, "a") as db:
            ds = db.create_dataset(
                allocated,
                data=small_void,
                maxshape=[None, 1],
                chunks=True,
                dtype="V10",
            )
            ds.attrs[COLUMN_NAMES_ATTRIBUTE] = ["col"]
            ds.attrs[BYTES_ATTRIBUTE] = True

        # Try appending data that would exceed the dataset's void type size
        large_void = np.array([np.void(b"\x00" * 20)]).reshape(1, 1)
        with pytest.raises(ValueError, match="exceeds existing dataset capacity"):
            _run(self.storage._write_raw_data(name, large_void, ["col"], is_bytes=True))

    def test_void_type_append_smaller_succeeds(self) -> None:
        """Appending serialized data with smaller/equal void type succeeds via cast."""
        name = "void_compat"
        allocated = PVCStorage.allocate_valid_dataset_name(name)
        filename = self.storage._get_filename(allocated)
        # Create dataset with V20
        data = np.array([np.void(b"\x00" * 20)]).reshape(1, 1)
        with h5py.File(filename, "a") as db:
            ds = db.create_dataset(
                allocated,
                data=data,
                maxshape=[None, 1],
                chunks=True,
                dtype="V20",
            )
            ds.attrs[COLUMN_NAMES_ATTRIBUTE] = ["col"]
            ds.attrs[BYTES_ATTRIBUTE] = True

        # Append data with V10 (smaller) -- should cast to V20 and succeed
        small_void = np.array([np.void(b"\x01" * 10)]).reshape(1, 1)
        _run(self.storage._write_raw_data(name, small_void, ["col"], is_bytes=True))

        # Verify 2 rows total
        assert _run(self.storage.dataset_rows(name)) == 2  # noqa: PLR2004 -- literal expected


# ===========================================================================
# Constructor edge cases
# ===========================================================================


class TestPVCConstructor:
    """Tests for PVCStorage constructor edge cases."""

    def test_nonexistent_directory(self) -> None:
        """PVCStorage can be created with a directory that doesn't exist yet."""
        nonexistent = Path(tempfile.gettempdir()) / (
            "nonexistent_trustyai_test_dir_" + uuid.uuid4().hex
        )
        storage = PVCStorage(str(nonexistent))
        assert storage.locks == {}

    def test_existing_directory_with_files(self) -> None:
        """PVCStorage discovers existing HDF5 files and creates locks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file matching the data_file pattern
            data_file = "test.hdf5"
            (Path(tmpdir) / f"model_a_{data_file}").touch()
            (Path(tmpdir) / f"model_b_{data_file}").touch()

            storage = PVCStorage(tmpdir, data_file=data_file)
            assert len(storage.locks) == 2  # noqa: PLR2004 -- literal expected


# ===========================================================================
# GlobalStorageInterface
# ===========================================================================


class TestGlobalStorageInterface:
    """Tests for the GlobalStorageInterface singleton."""

    def test_get_returns_pvc_by_default(self) -> None:
        """GlobalStorageInterface.get() returns PVCStorage by default."""
        from src.service.data.storage import GlobalStorageInterface  # noqa: PLC0415

        GlobalStorageInterface.reset()
        try:
            instance = GlobalStorageInterface.get()
            assert isinstance(instance, PVCStorage)
        finally:
            GlobalStorageInterface.reset()

    def test_reset_clears_singleton(self) -> None:
        """GlobalStorageInterface.reset() clears the cached instance."""
        from src.service.data.storage import GlobalStorageInterface  # noqa: PLC0415

        GlobalStorageInterface.reset()
        try:
            instance_1 = GlobalStorageInterface.get()
            GlobalStorageInterface.reset()
            instance_2 = GlobalStorageInterface.get()
            assert instance_1 is not instance_2
        finally:
            GlobalStorageInterface.reset()

    def test_force_reload(self) -> None:
        """GlobalStorageInterface.get(force_reload=True) creates a new instance."""
        from src.service.data.storage import GlobalStorageInterface  # noqa: PLC0415

        GlobalStorageInterface.reset()
        try:
            instance_1 = GlobalStorageInterface.get()
            instance_2 = GlobalStorageInterface.get(force_reload=True)
            assert instance_1 is not instance_2
        finally:
            GlobalStorageInterface.reset()

    def test_get_global_storage_interface_function(self) -> None:
        """get_global_storage_interface() delegates to GlobalStorageInterface.get()."""
        from src.service.data.storage import (  # noqa: PLC0415
            GlobalStorageInterface,
            get_global_storage_interface,
        )

        GlobalStorageInterface.reset()
        try:
            instance = get_global_storage_interface()
            assert isinstance(instance, PVCStorage)
        finally:
            GlobalStorageInterface.reset()


# ===========================================================================
# Void type exceeding MAX_VOID_TYPE_LENGTH in _write_raw_data
# ===========================================================================


class TestPVCVoidTypeMaxLength(_PVCFixture):
    """Tests for the MAX_VOID_TYPE_LENGTH guard in _write_raw_data."""

    def test_void_type_exceeds_max_raises(self) -> None:
        """Passing a void-typed array exceeding MAX_VOID_TYPE_LENGTH raises ValueError."""
        big_void = np.array([np.void(b"\x00" * (MAX_VOID_TYPE_LENGTH + 1))]).reshape(
            1, 1
        )
        with pytest.raises(ValueError, match="largest serializable void type"):
            _run(
                self.storage._write_raw_data(
                    "big_void_ds", big_void, ["col"], is_bytes=True
                )
            )


# ===========================================================================
# Partial payload & ModelMesh delegation
# ===========================================================================


class TestPVCPartialPayloads(_PVCFixture):
    """Tests for persist/get/delete partial payload edge cases and ModelMesh delegation."""

    def test_persist_and_get_modelmesh_payload(self) -> None:
        """persist_modelmesh_payload and get_modelmesh_payload delegate correctly."""
        from src.service.data.modelmesh_parser import PartialPayload  # noqa: PLC0415

        payload = PartialPayload(data="dGVzdA==")  # base64 for "test"
        _run(self.storage.persist_modelmesh_payload(payload, "mm-1", is_input=True))

        retrieved = _run(self.storage.get_modelmesh_payload("mm-1", is_input=True))
        assert retrieved is not None
        assert retrieved.data == payload.data

    def test_delete_modelmesh_payload(self) -> None:
        """delete_modelmesh_payload delegates to delete_partial_payload."""
        from src.service.data.modelmesh_parser import PartialPayload  # noqa: PLC0415

        payload = PartialPayload(data="dGVzdA==")
        _run(self.storage.persist_modelmesh_payload(payload, "mm-2", is_input=False))
        _run(self.storage.delete_modelmesh_payload("mm-2", is_input=False))

        result = _run(self.storage.get_modelmesh_payload("mm-2", is_input=False))
        assert result is None

    def test_delete_partial_payload_nonexistent_id_is_noop(self) -> None:
        """delete_partial_payload with a non-existent ID does nothing."""
        # No file exists, so this should be a no-op via MissingH5PYDataError path
        _run(self.storage.delete_partial_payload("ghost-id", is_input=True))

    def test_get_partial_payload_missing_file_returns_none(self) -> None:
        """get_partial_payload returns None when the HDF5 file doesn't exist."""
        result = _run(
            self.storage.get_partial_payload(
                "no-file-id", is_input=True, is_modelmesh=False
            )
        )
        assert result is None

    def test_delete_partial_payload_missing_id_in_existing_dataset(self) -> None:
        """delete_partial_payload with existing dataset but missing ID returns early."""
        from src.service.data.modelmesh_parser import PartialPayload  # noqa: PLC0415

        # Create the dataset by persisting one payload
        payload = PartialPayload(data="dGVzdA==")
        _run(self.storage.persist_partial_payload(payload, "exists-1", is_input=True))

        # Try deleting a non-existent ID in the same dataset
        _run(self.storage.delete_partial_payload("nonexistent-id", is_input=True))

        # Original payload should still be retrievable
        result = _run(
            self.storage.get_partial_payload(
                "exists-1", is_input=True, is_modelmesh=True
            )
        )
        assert result is not None


# ===========================================================================
# DeserializationError exception attributes
# ===========================================================================


class TestDeserializationError:
    """Tests for the DeserializationError exception class."""

    def test_str_includes_payload_id_and_reason(self) -> None:
        """Exception message includes payload ID and reason."""
        err = DeserializationError(
            payload_id="req-123",
            reason="bad format",
            original_exception=ValueError("oops"),
        )
        assert "req-123" in str(err)
        assert "bad format" in str(err)
        assert "ValueError" in str(err)

    def test_repr_includes_caused_by(self) -> None:
        """Repr includes caused_by info when original_exception is set."""
        err = DeserializationError(
            payload_id="req-456",
            reason="corrupt",
            original_exception=RuntimeError("boom"),
        )
        r = repr(err)
        assert "req-456" in r
        assert "RuntimeError" in r

    def test_repr_without_original_exception(self) -> None:
        """Repr works when no original_exception is provided."""
        err = DeserializationError(
            payload_id="req-789",
            reason="unknown",
        )
        r = repr(err)
        assert "req-789" in r
        assert "caused_by" not in r
