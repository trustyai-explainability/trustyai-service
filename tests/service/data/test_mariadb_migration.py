"""Tests for MoariaDB storage."""

import asyncio
import datetime
import unittest
from collections.abc import Coroutine
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mariadb")
from trustyai_service.service.data.storage.maria.maria import MariaDBStorage

alphabet = "abcdefghijklmnopqrstuvwxz"  # pragma: allowlist secret

# Test constants
MIN_MIGRATED_DATASETS = (
    12  # Minimum expected datasets after migration (4 models x 3 splits)
)


@pytest.mark.xdist_group("mariadb")
class TestMariaDBMigration(unittest.TestCase):
    """Test class for MariaDBStorage."""

    def setUp(self) -> None:
        """Set up MariaDB storage connection with migration enabled."""
        self.storage = MariaDBStorage(
            "trustyai",
            "trustyai",
            "127.0.0.1",
            3306,
            "trustyai-database",
            attempt_migration=True,
        )

    def tearDown(self) -> None:
        """Clean up MariaDB database after tests."""
        asyncio.run(self.storage.reset_database())

    async def _test_retrieve_data(self) -> None:
        # total data checks
        available_datasets = await self.storage.list_all_datasets()
        # Skip test if no legacy data is present (requires database to be
        # seeded with legacy_database_dump.sql)
        if "model1_inputs" not in available_datasets:
            self.skipTest(
                "No legacy data found - database must be seeded with "
                "tests/resources/legacy_database_dump.sql",
            )
        for i in [1, 2, 3, 4]:
            for split in ["inputs", "outputs", "metadata"]:
                assert f"model{i}_{split}" in available_datasets
        assert len(available_datasets) >= MIN_MIGRATED_DATASETS

        # model 1 checks
        model_1_inputs = await self.storage.read_data("model1_inputs", 0, 100)
        model_1_metadata = await self.storage.read_data("model1_metadata", 0, 1)
        assert np.array_equal(np.array([[0, 1, 2, 3, 4]] * 100), model_1_inputs)
        assert np.array_equal(np.array([0, 1, 2, 3, 4]), model_1_inputs[0])
        assert model_1_metadata[0][0] == datetime.datetime.fromisoformat(
            "2025-06-09 12:19:06.074828",
        )
        assert model_1_metadata[0][1] == "6decea54-91eb-4726-b6f8-a2f1dee8e81b"
        assert model_1_metadata[0][2] == "_trustyai_unlabeled"

        # model 3 checks
        assert await self.storage.get_aliased_column_names("model3_inputs") == [
            "year mapped",
            "make mapped",
            "color mapped",
        ]

        # model 4 checks
        model_4_inputs_row0 = await self.storage.read_data("model4_inputs", 0, 5)
        assert model_4_inputs_row0[0].tolist() == [0.0, "i'm text-0", True, 0]
        assert model_4_inputs_row0[4].tolist() == [4.0, "i'm text-4", True, 8]


def run_async_test(coro: Coroutine[Any, Any, None]) -> None:
    """Run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


TestMariaDBMigration.test_retrieve_data = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_retrieve_data(),
)


if __name__ == "__main__":
    unittest.main()
