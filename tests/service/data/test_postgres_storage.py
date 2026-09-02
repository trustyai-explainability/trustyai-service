"""Tests for PostgreSQL storage (real-DB integration).

These tests require a running PostgreSQL instance reachable at 127.0.0.1:5432.
Bring one up with:
    `podman compose -f tests/resources/compose-local-postgres.yaml up`
Without a database the tests error/skip, which is expected in CI without a DB.
"""

import asyncio
import unittest
from collections.abc import Coroutine
from typing import Any

import numpy as np
import pytest

pytest.importorskip("psycopg")
from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.data.storage.postgres.postgres import PostgreSQLStorage

alphabet = "abcdefghijklmnopqrstuvwxz"  # pragma: allowlist secret

# Test constants
EXPECTED_SINGLE_COLUMN_ROWS = 10  # Expected rows in single column dataset test


@pytest.mark.xdist_group("postgres")
class TestPostgreSQLStorage(unittest.TestCase):
    """Test class for PostgreSQLStorage."""

    def setUp(self) -> None:
        """Set up PostgreSQL storage connection for testing."""
        self.storage = PostgreSQLStorage(
            "trustyai",
            "trustyai",
            "127.0.0.1",
            5432,
            "trustyai-database",
        )
        self.original_datasets = set(asyncio.run(self.storage.list_all_datasets()))

    def tearDown(self) -> None:
        """Clean up PostgreSQL database after tests."""
        asyncio.run(self.storage.reset_database())

    async def _store_dataset(
        self,
        seed: int,
        n_rows: int | None = None,
        n_cols: int | None = None,
    ) -> tuple[np.ndarray, list[str], str]:
        """Create and store a test dataset with deterministic data based on seed."""
        n_rows = seed * 3 if n_rows is None else n_rows
        n_cols = seed + 10 if n_cols is None else n_cols
        dataset = np.arange(0, n_rows * n_cols).reshape(n_rows, n_cols)
        column_names = [alphabet[i] for i in range(dataset.shape[1])]
        dataset_name = f"dataset_{alphabet[seed]}"
        await self.storage.write_data(dataset_name, dataset, column_names)
        return dataset, column_names, dataset_name

    async def _test_retrieve_data(self) -> None:
        """Verify full and partial dataset retrieval from PostgreSQL storage."""
        for dataset_idx in range(1, 10):
            original_dataset, _, dataset_name = await self._store_dataset(dataset_idx)

            start_idx = dataset_idx
            n_rows = dataset_idx * 2
            retrieved_full_dataset = await self.storage.read_data(dataset_name)
            retrieved_partial_dataset = await self.storage.read_data(
                dataset_name,
                start_idx,
                n_rows,
            )

            assert np.array_equal(retrieved_full_dataset, original_dataset)
            assert original_dataset.shape == await self.storage.dataset_shape(
                dataset_name,
            )
            assert original_dataset.shape[0] == await self.storage.dataset_rows(
                dataset_name,
            )
            assert original_dataset.shape[1] == await self.storage.dataset_cols(
                dataset_name,
            )
            assert np.array_equal(
                retrieved_partial_dataset,
                original_dataset[start_idx : start_idx + n_rows],
            )

    async def _test_big_insert(self) -> None:
        """Verify storage and retrieval of large datasets with 5000 rows."""
        original_dataset, _, dataset_name = await self._store_dataset(0, 5000, 10)
        retrieved_full_dataset = await self.storage.read_data(dataset_name)

        assert np.array_equal(retrieved_full_dataset, original_dataset)
        assert original_dataset.shape == await self.storage.dataset_shape(dataset_name)
        assert original_dataset.shape[0] == await self.storage.dataset_rows(
            dataset_name,
        )
        assert original_dataset.shape[1] == await self.storage.dataset_cols(
            dataset_name,
        )

    async def _test_single_row_insert(self) -> None:
        """Verify storage and retrieval of single-row datasets."""
        original_dataset, _, dataset_name = await self._store_dataset(0, 1, 10)
        retrieved_full_dataset = await self.storage.read_data(dataset_name, 0, 1)

        assert np.array_equal(retrieved_full_dataset, original_dataset)
        assert original_dataset.shape == await self.storage.dataset_shape(dataset_name)
        assert original_dataset.shape[0] == await self.storage.dataset_rows(
            dataset_name,
        )
        assert original_dataset.shape[1] == await self.storage.dataset_cols(
            dataset_name,
        )

    async def _test_vector_retrieval(self) -> None:
        """Verify storage and retrieval of single-column vector datasets."""
        original_dataset = np.arange(0, 10)
        column_names = ["single_column"]
        dataset_name = "dataset_single_row"
        await self.storage.write_data(dataset_name, original_dataset, column_names)
        retrieved_full_dataset = await self.storage.read_data(dataset_name)
        transposed_dataset = retrieved_full_dataset.reshape(-1)

        assert np.array_equal(transposed_dataset, original_dataset)
        assert (
            await self.storage.dataset_rows(dataset_name) == EXPECTED_SINGLE_COLUMN_ROWS
        )
        assert await self.storage.dataset_cols(dataset_name) == 1

    async def _test_list_all_datasets(self) -> None:
        """Verify list_all_datasets returns all written dataset names."""
        stored_names = set()
        for dataset_idx in range(1, 5):
            _, _, dataset_name = await self._store_dataset(dataset_idx)
            stored_names.add(dataset_name)

        listed = set(await self.storage.list_all_datasets()) - self.original_datasets
        assert listed == stored_names

    async def _test_name_mapping(self) -> None:
        """Verify column name aliasing and original name preservation."""
        for dataset_idx in range(1, 10):
            _, column_names, dataset_name = await self._store_dataset(dataset_idx)
            name_mapping = {
                name: "aliased_" + name
                for i, name in enumerate(column_names)
                if i % 2 == 0
            }
            expected_mapping = [name_mapping.get(name, name) for name in column_names]
            await self.storage.apply_name_mapping(dataset_name, name_mapping)

            retrieved_original_names = await self.storage.get_original_column_names(
                dataset_name,
            )
            retrieved_aliased_names = await self.storage.get_aliased_column_names(
                dataset_name,
            )

            assert column_names == retrieved_original_names
            assert expected_mapping == retrieved_aliased_names

    async def _test_clear_name_mapping(self) -> None:
        """Verify clearing a name mapping resets aliased names to originals."""
        _, column_names, dataset_name = await self._store_dataset(3)
        name_mapping = {column_names[0]: "aliased_" + column_names[0]}
        await self.storage.apply_name_mapping(dataset_name, name_mapping)
        await self.storage.clear_name_mapping(dataset_name)

        retrieved_aliased_names = await self.storage.get_aliased_column_names(
            dataset_name,
        )
        assert column_names == retrieved_aliased_names

    async def _test_delete_dataset(self) -> None:
        """Verify a dataset can be deleted."""
        _, _, dataset_name = await self._store_dataset(2)
        assert await self.storage.dataset_exists(dataset_name)
        await self.storage.delete_dataset(dataset_name)
        assert not await self.storage.dataset_exists(dataset_name)

    async def _test_partial_payload(self) -> None:
        """Verify partial-payload persist / get / delete roundtrip."""
        payload_id = "req-123"
        payload = PartialPayload(data="dGVzdA==")  # base64 for "test"

        await self.storage.persist_partial_payload(payload, payload_id, is_input=True)
        retrieved = await self.storage.get_partial_payload(
            payload_id, is_input=True, is_modelmesh=True
        )
        assert retrieved is not None
        assert retrieved.data == payload.data

        await self.storage.delete_partial_payload(payload_id, is_input=True)
        assert (
            await self.storage.get_partial_payload(
                payload_id, is_input=True, is_modelmesh=True
            )
            is None
        )

    def test_retrieve_data(self) -> None:
        """Test full and partial dataset retrieval."""
        run_async_test(self._test_retrieve_data())

    def test_name_mapping(self) -> None:
        """Test column name aliasing functionality."""
        run_async_test(self._test_name_mapping())

    def test_clear_name_mapping(self) -> None:
        """Test clearing column name aliases."""
        run_async_test(self._test_clear_name_mapping())

    def test_list_all_datasets(self) -> None:
        """Test listing all stored datasets."""
        run_async_test(self._test_list_all_datasets())

    def test_delete_dataset(self) -> None:
        """Test deleting a dataset."""
        run_async_test(self._test_delete_dataset())

    def test_partial_payload(self) -> None:
        """Test partial-payload persist/get/delete."""
        run_async_test(self._test_partial_payload())

    def test_big_insert(self) -> None:
        """Test large dataset storage and retrieval."""
        run_async_test(self._test_big_insert())

    def test_single_row_insert(self) -> None:
        """Test single-row dataset storage and retrieval."""
        run_async_test(self._test_single_row_insert())

    def test_single_row_retrieval(self) -> None:
        """Test single-column vector dataset storage and retrieval."""
        run_async_test(self._test_vector_retrieval())


def run_async_test(coro: Coroutine[Any, Any, None]) -> None:
    """Run async tests."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    unittest.main()
