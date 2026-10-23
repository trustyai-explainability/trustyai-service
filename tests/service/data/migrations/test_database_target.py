"""Unit tests for the dialect-neutral database target."""

from __future__ import annotations

import numpy as np
import pytest


class _FakeDuplicateTable:
    """Mimic a MariaDB 'table already exists' driver error."""

    errno = 1050


@pytest.mark.asyncio
async def test_database_target_rejects_incompatible_collision(
    migration_api, sqlite_target
):
    """A schema collision must leave existing destination rows untouched."""
    await sqlite_target.storage.write_data("existing", np.array([[7]]), ["original"])
    descriptor = migration_api.DatasetDescriptor(
        name="existing",
        column_names=("different",),
        aliases=("different",),
        column_types=("DOUBLE",),
        row_count=1,
    )
    with pytest.raises(ValueError, match="incompatible column names"):
        await sqlite_target.ensure_dataset(descriptor)
    assert (await sqlite_target.storage.read_data("existing")).tolist() == [[7]]


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
async def test_database_target_preserves_bytes_and_mapping_cells(sqlite_target) -> None:
    """Canonical reads retain scalar bytes and mappings as Python values."""
    rows = np.empty((2, 2), dtype=object)
    rows[:, 0] = [b"\x00\x01", b"\x02\x03"]
    rows[:, 1] = [{"rank": 1}, {"rank": 2}]
    storage = sqlite_target.storage

    await storage.write_data("typed_cells", rows, ["raw", "details"])
    actual = await storage.read_data("typed_cells")

    assert actual.tolist() == rows.tolist()
    assert isinstance(actual[0, 0], bytes)
    assert isinstance(actual[0, 1], dict)


@pytest.mark.asyncio
@pytest.mark.parametrize("column_count", [1, 2])
async def test_database_target_keeps_tensor_cells_in_the_declared_matrix(
    sqlite_target, column_count
):
    """Equal-sized tensor cells must not become extra dataset dimensions."""
    rows = np.empty((2, column_count), dtype=object)
    for index in range(2):
        for column in range(column_count):
            rows[index, column] = [1.25, 2.5]
    names = [f"tensor_{column}" for column in range(column_count)]
    await sqlite_target.storage.write_data("tensors", rows, names)
    actual = await sqlite_target.storage.read_data("tensors")
    assert actual.shape == (2, column_count)
    assert actual.tolist() == rows.tolist()


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
        await first.storage.reset_for_test()
        await first.checkpoint_storage.reset_for_test()
        await second.checkpoint_storage.reset_for_test()
        values = {
            "migration_id": "run",
            "source_type": "PVC",
            "source_identity": "digest",
            "status": "IN_PROGRESS",
            "total_files": 1,
            "total_datasets": 1,
            "total_rows": 2,
            "error_message": None,
        }
        await first.save_migration_run(values)
        await second.save_migration_run(values)

        await first.checkpoint_storage.reset_for_test()

        assert await first.get_migration_run("run") is None
        assert await second.get_migration_run("run") == values
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_database_target_retries_concurrent_table_creation(
    sqlite_target,
) -> None:
    """A duplicate-object race while creating tables is retried, not fatal."""
    from sqlalchemy.exc import DBAPIError

    storage = sqlite_target.storage
    real_create_all = storage.tables.metadata.create_all
    calls = []
    race_statement = "CREATE TABLE trusty_migration_race_datasets"

    def flaky_create_all(bind) -> None:
        calls.append(bind)
        if len(calls) == 1:
            raise DBAPIError(race_statement, (), _FakeDuplicateTable())
        real_create_all(bind)

    storage.tables.metadata.create_all = flaky_create_all
    try:
        await storage.initialize()
    finally:
        del storage.tables.metadata.create_all

    assert len(calls) == 2
    await storage.write_data(
        "race_inputs",
        np.asarray([[1.0]], dtype=np.float64),
        ["value"],
    )
    assert await storage.dataset_rows("race_inputs") == 1


@pytest.mark.asyncio
async def test_database_target_does_not_mask_non_ddl_failures(
    sqlite_target,
) -> None:
    """Driver errors that are not duplicate-table races still fail loudly."""
    from sqlalchemy.exc import DBAPIError

    storage = sqlite_target.storage
    calls = []
    insert_statement = "INSERT INTO trusty_migration_race_datasets"

    def failing_create_all(bind) -> None:
        calls.append(bind)
        raise DBAPIError(insert_statement, (), _FakeDuplicateTable())

    storage.tables.metadata.create_all = failing_create_all
    try:
        with pytest.raises(DBAPIError):
            await storage.initialize()
    finally:
        del storage.tables.metadata.create_all

    assert len(calls) == 1
