"""
Tests for MoariaDB storage
"""

import asyncio
import datetime
import unittest
import numpy as np

from src.service.data.storage.maria.maria import MariaDBStorage


alphabet = "abcdefghijklmnopqrstuvwxz"

class TestMariaDBMigration(unittest.TestCase):
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
            attempt_migration=True)


    def tearDown(self):
        asyncio.run(self.storage.reset_database())


    async def _test_retrieve_data(self):
        # total data checks
        available_datasets = self.storage.list_all_datasets()
        self.assertEqual(len(available_datasets), 12)

        # model 1 checks
        model_1_inputs = await self.storage.read_data("model1_inputs", 0, 100)
        model_1_metadata = await self.storage.read_data("model1_metadata", 0, 1)
        self.assertTrue(np.array_equal(np.array([[0,1,2,3,4]]*100), model_1_inputs))
        self.assertTrue(np.array_equal(np.array([0, 1, 2, 3, 4]), model_1_inputs[0]))
        self.assertEqual(model_1_metadata[0][0], datetime.datetime.fromisoformat("2025-06-09 12:19:06.074828"))
        self.assertEqual(model_1_metadata[0][1], "6decea54-91eb-4726-b6f8-a2f1dee8e81b")
        self.assertEqual(model_1_metadata[0][2], "_trustyai_unlabeled")

        # model 3 checks
        self.assertEqual(await self.storage.get_aliased_column_names("model3_inputs"), ["year mapped", "make mapped", "color mapped"])

        # model 4 checks
        model_4_inputs_row0 = await self.storage.read_data("model4_inputs", 0, 5)
        self.assertEqual(model_4_inputs_row0[0].tolist(), [0.0, "i'm text-0", True, 0])
        self.assertEqual(model_4_inputs_row0[4].tolist(), [4.0, "i'm text-4", True, 8])




def run_async_test(coro):
    """Helper function to run async tests."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


TestMariaDBMigration.test_retrieve_data = lambda self: run_async_test(
    self._test_retrieve_data()
)


if __name__ == "__main__":
    unittest.main()