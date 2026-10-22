"""Atomic batch-write regressions shared by every available SQL backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import event

from tests.service.data.storage import test_sql_parity as parity
from tests.service.data.storage.test_sql_parity import _inference_batch

if TYPE_CHECKING:
    from sqlalchemy import Connection

    from tests.service.data.storage.sql_test_resources import SQLTestResources
    from trustyai_service.service.data.storage.sql.base import SQLStorage


# Keep the same backend parametrization and ownership-based fixture cleanup.
names = parity.names
storage = parity.storage


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_dataset", [0, 1, 2])
async def test_batch_preflight_rejects_without_any_writes(
    storage: SQLStorage, names: SQLTestResources, invalid_dataset: int
) -> None:
    """A schema rejection in any position leaves all three datasets unchanged."""
    dataset_names = [
        names.dataset(suffix) for suffix in ("inputs", "outputs", "metadata")
    ]
    original = _inference_batch(dataset_names, 1, "first_0")
    await storage.write_data_batch(original)
    rejected = _inference_batch(dataset_names, 2, "rejected_0")
    dataset, data, _ = rejected[invalid_dataset]
    rejected[invalid_dataset] = (dataset, data, ["renamed"])
    mutations = []

    def capture_mutations(
        _conn: Connection, _cursor: object, statement: str, *_args: object
    ) -> None:
        if statement.lstrip().split()[0].upper() in {
            "INSERT",
            "UPDATE",
            "CREATE",
            "DELETE",
        }:
            mutations.append(statement)

    event.listen(storage._engine, "before_cursor_execute", capture_mutations)
    try:
        with pytest.raises(ValueError, match="Column mismatch"):
            await storage.write_data_batch(rejected)
    finally:
        event.remove(storage._engine, "before_cursor_execute", capture_mutations)

    assert mutations == []
    for dataset, data, _ in original:
        assert await storage.dataset_rows(dataset) == 1
        np.testing.assert_array_equal(await storage.read_data(dataset), data)


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_datasets", [0, 1, 3])
@pytest.mark.parametrize("failed_dataset", [1, 2])
async def test_batch_insert_failure_rolls_back_all_rows(
    storage: SQLStorage,
    names: SQLTestResources,
    existing_datasets: int,
    failed_dataset: int,
) -> None:
    """Later insert failures roll back rows for new, mixed, and existing batches."""
    dataset_names = [
        names.dataset(suffix) for suffix in ("inputs", "outputs", "metadata")
    ]
    original = _inference_batch(dataset_names, 1, "first_0")
    for dataset, data, columns in original[:existing_datasets]:
        await storage.write_data(dataset, data, columns)
    inserted_datasets = []

    def fail_before_insert(
        _conn: Connection, _cursor: object, statement: str, *_args: object
    ) -> None:
        if statement.startswith("INSERT INTO trustyai_v2_dataset_"):
            inserted_datasets.append(statement)
            if len(inserted_datasets) == failed_dataset + 1:
                msg = "simulated batch insert failure"
                raise RuntimeError(msg)

    event.listen(storage._engine, "before_cursor_execute", fail_before_insert)
    try:
        with pytest.raises(RuntimeError, match="simulated batch insert failure"):
            await storage.write_data_batch(
                _inference_batch(dataset_names, 2, "rejected_0")
            )
    finally:
        event.remove(storage._engine, "before_cursor_execute", fail_before_insert)

    assert len(inserted_datasets) == failed_dataset + 1
    for index, (dataset, data, _) in enumerate(original):
        if index < existing_datasets:
            assert await storage.dataset_rows(dataset) == 1
            np.testing.assert_array_equal(await storage.read_data(dataset), data)
        elif await storage.dataset_exists(dataset):
            # MariaDB commits CREATE TABLE separately, but no inference rows.
            assert await storage.dataset_rows(dataset) == 0
            assert (await storage.read_data(dataset)).size == 0

    accepted = _inference_batch(dataset_names, 3, "next_0")
    await storage.write_data_batch(accepted)
    for index, (dataset, data, _) in enumerate(accepted):
        expected = (
            np.vstack([original[index][1], data]) if index < existing_datasets else data
        )
        assert await storage.dataset_rows(dataset) == len(expected)
        np.testing.assert_array_equal(await storage.read_data(dataset), expected)
