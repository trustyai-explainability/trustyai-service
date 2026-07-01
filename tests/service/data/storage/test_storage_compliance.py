"""StorageInterface compliance test suite.

Every StorageInterface implementation must pass these tests.
Backend-specific subclasses provide the `storage` fixture.
"""

import inspect

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from src.endpoints.consumer import KServeData, KServeInferenceRequest
from src.service.data.storage.db.db_storage import DBStorage
from src.service.data.storage.pvc import PVCStorage


class StorageComplianceTests:
    """Contract tests for StorageInterface implementations."""

    # --- dataset lifecycle ---

    @pytest.mark.asyncio
    async def test_dataset_exists_false_initially(self, storage) -> None:
        """Non-existent dataset returns False."""
        assert not await storage.dataset_exists("nonexistent")

    @pytest.mark.asyncio
    async def test_write_creates_dataset(self, storage) -> None:
        """Writing data creates the dataset."""
        data = np.array([[1.0, 2.0]])
        await storage.write_data("ds", data, ["a", "b"])
        assert await storage.dataset_exists("ds")

    @pytest.mark.asyncio
    async def test_read_roundtrip_matches_write(self, storage) -> None:
        """Data read back matches what was written."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        await storage.write_data("ds", data, ["a", "b"])
        result = await storage.read_data("ds")
        assert np.array_equal(result, data)

    @pytest.mark.asyncio
    async def test_read_partial_with_offset_and_limit(self, storage) -> None:
        """Partial read with offset and limit returns the correct slice."""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        await storage.write_data("ds", data, ["x"])
        result = await storage.read_data("ds", start_row=1, n_rows=2)
        expected = np.array([[2.0], [3.0]])
        assert np.array_equal(result, expected)

    @pytest.mark.asyncio
    async def test_dataset_rows_matches_written(self, storage) -> None:
        """Row count matches the number of rows written."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        await storage.write_data("ds", data, ["a", "b"])
        assert await storage.dataset_rows("ds") == 2  # noqa: PLR2004

    @pytest.mark.asyncio
    async def test_dataset_shape_matches_written(self, storage) -> None:
        """Shape matches the written data dimensions."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        await storage.write_data("ds", data, ["a", "b", "c"])
        shape = await storage.dataset_shape("ds")
        assert shape == (2, 3)

    @pytest.mark.asyncio
    async def test_append_data_increments_rows(self, storage) -> None:
        """Appending data increases the row count."""
        data1 = np.array([[1.0, 2.0]])
        data2 = np.array([[3.0, 4.0], [5.0, 6.0]])
        await storage.write_data("ds", data1, ["a", "b"])
        await storage.write_data("ds", data2, ["a", "b"])
        assert await storage.dataset_rows("ds") == 3  # noqa: PLR2004
        result = await storage.read_data("ds")
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert np.array_equal(result, expected)

    @pytest.mark.asyncio
    async def test_append_shape_mismatch_raises(self, storage) -> None:
        """Appending data with wrong column count raises ValueError."""
        data1 = np.array([[1.0, 2.0]])
        data2 = np.array([[3.0, 4.0, 5.0]])
        await storage.write_data("ds", data1, ["a", "b"])
        with pytest.raises(ValueError, match=r"[Ss]hape"):
            await storage.write_data("ds", data2, ["a", "b", "c"])

    @pytest.mark.asyncio
    async def test_delete_dataset_removes_it(self, storage) -> None:
        """Deleting a dataset removes it from storage."""
        data = np.array([[1.0]])
        await storage.write_data("ds", data, ["x"])
        await storage.delete_dataset("ds")
        assert not await storage.dataset_exists("ds")

    @pytest.mark.asyncio
    async def test_list_all_datasets(self, storage) -> None:
        """Listing datasets returns all written datasets."""
        await storage.write_data("ds_a", np.array([[1.0]]), ["x"])
        await storage.write_data("ds_b", np.array([[2.0]]), ["y"])
        datasets = await storage.list_all_datasets()
        assert set(datasets) >= {"ds_a", "ds_b"}

    # --- column name mapping ---

    @pytest.mark.asyncio
    async def test_original_names_match_write(self, storage) -> None:
        """Original column names match what was written."""
        await storage.write_data("ds", np.array([[1.0, 2.0]]), ["col_a", "col_b"])
        names = await storage.get_original_column_names("ds")
        assert list(names) == ["col_a", "col_b"]

    @pytest.mark.asyncio
    async def test_aliased_names_default_to_original(self, storage) -> None:
        """Aliased names default to original names before mapping."""
        await storage.write_data("ds", np.array([[1.0]]), ["col_x"])
        aliased = await storage.get_aliased_column_names("ds")
        assert list(aliased) == ["col_x"]

    @pytest.mark.asyncio
    async def test_apply_name_mapping(self, storage) -> None:
        """Applying name mapping changes aliased names."""
        await storage.write_data("ds", np.array([[1.0, 2.0]]), ["a", "b"])
        await storage.apply_name_mapping("ds", {"a": "Alpha"})
        aliased = await storage.get_aliased_column_names("ds")
        assert list(aliased) == ["Alpha", "b"]
        original = await storage.get_original_column_names("ds")
        assert list(original) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_clear_name_mapping_reverts(self, storage) -> None:
        """Clearing name mapping reverts to original names."""
        await storage.write_data("ds", np.array([[1.0]]), ["col"])
        await storage.apply_name_mapping("ds", {"col": "Column One"})
        await storage.clear_name_mapping("ds")
        aliased = await storage.get_aliased_column_names("ds")
        assert list(aliased) == ["col"]

    # --- model discovery ---

    @pytest.mark.asyncio
    async def test_get_known_models_empty(self, storage) -> None:
        """No known models when storage is empty."""
        models = await storage.get_known_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_get_known_models_from_suffixed_datasets(self, storage) -> None:
        """Known models are extracted from suffixed dataset names."""
        await storage.write_data("model_A_inputs", np.array([[1.0]]), ["x"])
        await storage.write_data("model_A_outputs", np.array([[0.0]]), ["y"])
        await storage.write_data("model_B_inputs", np.array([[2.0]]), ["x"])
        models = await storage.get_known_models()
        assert set(models) == {"model_A", "model_B"}

    # --- partial payloads (KServe reconciliation) ---

    @pytest.mark.asyncio
    async def test_persist_and_retrieve_partial_payload(self, storage) -> None:
        """Persisted partial payload can be retrieved."""
        req = KServeInferenceRequest(
            id="req-1",
            inputs=[KServeData(name="x", shape=[1], datatype="FP64", data=[1.0])],
        )
        await storage.persist_partial_payload(req, "req-1", is_input=True)
        retrieved = await self._get_partial(storage, "req-1", is_input=True)
        assert retrieved is not None
        assert retrieved.id == "req-1"

    @pytest.mark.asyncio
    async def test_delete_partial_payload(self, storage) -> None:
        """Deleted partial payload is no longer retrievable."""
        req = KServeInferenceRequest(
            id="req-2",
            inputs=[KServeData(name="x", shape=[1], datatype="FP64", data=[2.0])],
        )
        await storage.persist_partial_payload(req, "req-2", is_input=True)
        await storage.delete_partial_payload("req-2", is_input=True)
        result = await self._get_partial(storage, "req-2", is_input=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_missing_partial_returns_none(self, storage) -> None:
        """Retrieving a non-existent partial payload returns None."""
        result = await self._get_partial(storage, "nonexistent", is_input=True)
        assert result is None

    async def _get_partial(self, storage, payload_id: str, *, is_input: bool) -> object:
        """Call get_partial_payload with backward compat for is_modelmesh."""
        sig = inspect.signature(storage.get_partial_payload)
        if "is_modelmesh" in sig.parameters:
            return await storage.get_partial_payload(
                payload_id, is_input=is_input, is_modelmesh=False
            )
        return await storage.get_partial_payload(payload_id, is_input=is_input)

    # --- edge cases ---

    @pytest.mark.asyncio
    async def test_write_single_row(self, storage) -> None:
        """Single-row dataset can be written and read."""
        data = np.array([[42.0]])
        await storage.write_data("ds", data, ["val"])
        result = await storage.read_data("ds")
        assert np.array_equal(result, data)

    @pytest.mark.asyncio
    async def test_write_single_column_vector(self, storage) -> None:
        """1D vector is reshaped to column and stored correctly."""
        data = np.arange(5)
        await storage.write_data("ds", data, ["val"])
        result = await storage.read_data("ds")
        assert result.shape[0] == 5  # noqa: PLR2004
        assert await storage.dataset_rows("ds") == 5  # noqa: PLR2004

    @pytest.mark.asyncio
    async def test_write_non_numeric_data(self, storage) -> None:
        """Non-numeric (string) data can be stored and retrieved."""
        data = np.array([["hello", "world"], ["foo", "bar"]], dtype="O")
        await storage.write_data("ds", data, ["a", "b"])
        result = await storage.read_data("ds")
        assert result[0][0] == "hello"
        assert result[1][1] == "bar"

    @pytest.mark.asyncio
    async def test_empty_write_raises(self, storage) -> None:
        """Writing empty data raises ValueError (DBStorage) or is a no-op (PVC)."""
        if isinstance(storage, DBStorage):
            with pytest.raises(ValueError, match=r"[Nn]o data"):
                await storage.write_data("ds", np.array([]), ["x"])
        else:
            pytest.skip("PVC does not raise on empty write")


# === Backend-specific subclasses ===


class TestPVCCompliance(StorageComplianceTests):
    """PVC (HDF5) storage compliance tests."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Provide a PVCStorage instance."""
        return PVCStorage(str(tmp_path))


class TestDBStorageSQLite(StorageComplianceTests):
    """DBStorage with SQLite in-memory backend."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Provide a DBStorage instance with SQLite."""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        s = DBStorage(engine)
        await s.initialize()
        yield s
        await engine.dispose()
