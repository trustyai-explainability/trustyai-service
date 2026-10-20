"""Unit tests for the dialect-neutral database target."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_database_target_preserves_rows_and_column_metadata(
    sqlite_target,
) -> None:
    """The target stores logical rows independently of SQL dialect."""
    storage = sqlite_target.storage
    rows = np.asarray([[1.0, "one"], [2.0, "two"]], dtype=object)

    await storage.write_data("unit_inputs", rows, ["number", "label"])
    await storage.set_column_types("unit_inputs", ["DOUBLE", "STRING"])
    await storage.apply_name_mapping("unit_inputs", {"label": "Label"})

    np.testing.assert_array_equal(await storage.read_data("unit_inputs"), rows)
    assert await storage.get_original_column_names("unit_inputs") == [
        "number",
        "label",
    ]
    assert await storage.get_aliased_column_names("unit_inputs") == [
        "number",
        "Label",
    ]
    assert await storage.get_column_types("unit_inputs") == ["DOUBLE", "STRING"]


@pytest.mark.asyncio
async def test_database_target_preserves_serialized_metadata_on_append(
    sqlite_target,
) -> None:
    """A generic row append must not erase source serialization metadata."""
    from sqlalchemy import select

    storage = sqlite_target.storage
    await storage.ensure_dataset("serialized_inputs", ["payload"], serialized=True)
    await storage.write_data(
        "serialized_inputs",
        np.asarray([["value"]], dtype=object),
        ["payload"],
    )

    async with storage.engine.connect() as connection:
        result = await connection.execute(
            select(storage.tables.datasets.c.metadata).where(
                storage.tables.datasets.c.dataset_name == "serialized_inputs"
            )
        )
        assert result.scalar_one()["serialized"] is True


@pytest.mark.asyncio
async def test_database_target_persists_partial_payloads(
    sqlite_target, partial_payload
):
    """KServe reconciliation payloads use the same target namespace."""
    storage = sqlite_target.storage

    await storage.persist_partial_payload(
        partial_payload,
        "pending-1",
        is_input=True,
    )

    assert (
        await storage.get_partial_payload(
            "pending-1",
            is_input=True,
            is_modelmesh=False,
        )
        == partial_payload
    )


@pytest.mark.asyncio
async def test_database_target_reset_is_namespace_scoped(migration_api):
    """Test cleanup cannot remove another migration namespace."""
    pytest.importorskip("aiosqlite")
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    first = migration_api.DatabaseTarget(engine, migration_namespace="first")
    second = migration_api.DatabaseTarget(engine, migration_namespace="second")
    try:
        await first.reset_for_test()
        await second.reset_for_test()
        await first.storage.write_data(
            "first_dataset",
            np.asarray([[1.0]], dtype=np.float64),
            ["value"],
        )
        await second.storage.write_data(
            "second_dataset",
            np.asarray([[2.0]], dtype=np.float64),
            ["value"],
        )

        await first.reset_for_test()

        assert await first.storage.list_all_datasets() == []
        assert await second.storage.list_all_datasets() == ["second_dataset"]
    finally:
        await engine.dispose()
