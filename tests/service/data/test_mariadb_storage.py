"""
Tests for MoariaDB storage
"""

import asyncio
import unittest
import os
import numpy as np

from src.service.data.storage.maria.maria import MariaDBStorage


alphabet = "abcdefghijklmnopqrstuvwxz"

class TestMariaDBStorage(unittest.TestCase):
    """
    Test class for MariaDBStorage
    """

    def setUp(self):
        self.storage = MariaDBStorage(
            "trustyai",
            "trustyai",
            "127.0.0.1",
            3306,
            "trustyai-database",
            attempt_migration=False)


    def tearDown(self):
        self.storage.reset_database()

    async def _store_dataset(self, seed, n_rows=None, n_cols=None):
        n_rows = seed * 3 if n_rows is None else n_rows
        n_cols = seed + 10 if n_cols is None else n_cols
        dataset = np.arange(0, n_rows * n_cols).reshape(n_rows, n_cols)
        column_names = [alphabet[i] for i in range(dataset.shape[1])]
        dataset_name = f"dataset_{alphabet[seed]}"
        await self.storage.write_data(dataset_name, dataset, column_names)
        return dataset, column_names, dataset_name

    async def _test_retrieve_data(self):
        for dataset_idx in range(1, 10):
            original_dataset, _, dataset_name = await self._store_dataset(dataset_idx)

            start_idx = dataset_idx
            n_rows =  dataset_idx * 2
            retrieved_full_dataset = self.storage.read_data(dataset_name)
            retrieved_partial_dataset = self.storage.read_data(dataset_name, start_idx, n_rows)

            self.assertTrue(np.array_equal(retrieved_full_dataset, original_dataset))
            self.assertEqual(original_dataset.shape, self.storage.dataset_shape(dataset_name))
            self.assertEqual(original_dataset.shape[0], self.storage.dataset_rows(dataset_name))
            self.assertEqual(original_dataset.shape[1], self.storage.dataset_cols(dataset_name))
            self.assertTrue(np.array_equal(retrieved_partial_dataset, original_dataset[start_idx:start_idx+n_rows]))

    async def _test_big_insert(self):
        original_dataset, _, dataset_name = await self._store_dataset(0,  5000, 10)
        retrieved_full_dataset = self.storage.read_data(dataset_name)

        self.assertTrue(np.array_equal(retrieved_full_dataset, original_dataset))
        self.assertEqual(original_dataset.shape, self.storage.dataset_shape(dataset_name))
        self.assertEqual(original_dataset.shape[0], self.storage.dataset_rows(dataset_name))
        self.assertEqual(original_dataset.shape[1], self.storage.dataset_cols(dataset_name))

    async def _test_single_row_insert(self):
        original_dataset, _, dataset_name = await self._store_dataset(0,  1, 10)
        retrieved_full_dataset = self.storage.read_data(dataset_name, 0, 1)

        self.assertTrue(np.array_equal(retrieved_full_dataset, original_dataset))
        self.assertEqual(original_dataset.shape, self.storage.dataset_shape(dataset_name))
        self.assertEqual(original_dataset.shape[0], self.storage.dataset_rows(dataset_name))
        self.assertEqual(original_dataset.shape[1], self.storage.dataset_cols(dataset_name))


    async def _test_single_row_retrieval(self):
        original_dataset = np.arange(0, 10).reshape(1, 10)
        column_names = [alphabet[i] for i in range(original_dataset.shape[0])]
        dataset_name = "dataset_single_row"
        await self.storage.write_data(dataset_name, original_dataset, column_names)
        retrieved_full_dataset = self.storage.read_data(dataset_name)[0]

        self.assertTrue(np.array_equal(retrieved_full_dataset, original_dataset))
        self.assertEqual(1, self.storage.dataset_rows(dataset_name))
        self.assertEqual(10, self.storage.dataset_cols(dataset_name))

    async def _test_name_mapping(self):
        for dataset_idx in range(1, 10):
            original_dataset, column_names, dataset_name = await self._store_dataset(dataset_idx)
            name_mapping = {name: "aliased_" + name for i, name in enumerate(column_names) if i % 2 == 0}
            expected_mapping = [name_mapping.get(name, name) for name in column_names]
            self.storage.apply_name_mapping(dataset_name, name_mapping)

            self.assertEqual(column_names, self.storage.get_original_column_names(dataset_name))
            self.assertEqual(expected_mapping, self.storage.get_aliased_column_names(dataset_name))

def run_async_test(coro):
    """Helper function to run async tests."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


TestMariaDBStorage.test_retrieve_data = lambda self: run_async_test(
    self._test_retrieve_data()
)
TestMariaDBStorage.test_name_mapping = lambda self: run_async_test(
    self._test_name_mapping()
)

TestMariaDBStorage.test_big_insert = lambda self: run_async_test(
    self._test_big_insert()
)

TestMariaDBStorage.test_single_row_insert = lambda self: run_async_test(
    self._test_single_row_insert()
)

TestMariaDBStorage._test_single_row_retrieval = lambda self: run_async_test(
    self._test_single_row_retrieval()
)




if __name__ == "__main__":
    unittest.main()