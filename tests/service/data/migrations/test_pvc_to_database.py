"""End-to-end contract tests for PVC to MariaDB and PostgreSQL."""

from __future__ import annotations

import pytest

from .conftest import (
    assert_dataset_matches,
    result_status,
    target_storage,
)
from .test_migration_contract import _runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_kind",
    ["mariadb", "postgres"],
    ids=["maria", "postgres"],
)
async def test_pvc_to_each_database_preserves_data_and_metadata(
    migration_api, pvc_fixture, target_kind, target_factory
) -> None:
    """Both SQL targets receive the same logical datasets and metadata."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory(target_kind)

    result = await _runner(migration_api, source, target, batch_size=2).run()

    assert result_status(result) == "COMPLETE"
    assert result.datasets_written == 3
    assert result.rows_written == 6
    for expected in pvc_fixture.datasets:
        await assert_dataset_matches(target, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["mariadb", "postgres"])
async def test_pvc_to_each_database_respects_batch_boundaries(
    migration_api, pvc_fixture, target_kind, target_factory
) -> None:
    """Large logical datasets are streamed in bounded batches."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory(target_kind)
    runner = _runner(migration_api, source, target, batch_size=1)

    result = await runner.run()

    assert result_status(result) == "COMPLETE"
    assert result.max_batch_rows <= 1
    assert result.rows_read == result.rows_written == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["mariadb", "postgres"])
async def test_pvc_to_each_database_decodes_serialized_hdf5_rows(
    migration_api, pvc_fixture, target_kind, target_factory
) -> None:
    """JSON+gzip HDF5 rows are restored as bytes, booleans, and nested values."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory(target_kind)
    await _runner(migration_api, source, target).run()

    encoded = pvc_fixture.datasets[1]
    actual = await target_storage(target).read_data(encoded.name)
    assert [tuple(row.tolist()) for row in actual] == [
        tuple(row) for row in encoded.rows
    ]
    assert isinstance(actual[0][2], bytes)
    assert isinstance(actual[0][3], dict)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["mariadb", "postgres"])
async def test_pvc_to_each_database_retains_empty_dataset_metadata(
    migration_api, pvc_fixture, target_kind, target_factory
) -> None:
    """An empty source dataset remains discoverable with its schema."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory(target_kind)
    await _runner(migration_api, source, target).run()

    empty = pvc_fixture.datasets[2]
    storage = target_storage(target)
    assert await storage.dataset_exists(empty.name)
    assert await storage.dataset_rows(empty.name) == 0
    assert await storage.get_original_column_names(empty.name) == list(
        empty.column_names
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["mariadb", "postgres"])
async def test_pvc_to_each_database_preserves_partial_payloads(
    migration_api,
    pvc_fixture_with_partial_payload,
    target_kind,
    target_factory,
    partial_payload,
) -> None:
    """In-flight reconciliation payloads are copied with inference data."""
    source = migration_api.PVCSource(
        pvc_fixture_with_partial_payload.directory,
        quiesced=True,
    )
    target = await target_factory(target_kind)

    result = await _runner(migration_api, source, target).run()

    assert result_status(result) == "COMPLETE"
    migrated = await target_storage(target).get_partial_payload(
        "pending-1", is_input=True, is_modelmesh=False
    )
    assert migrated == partial_payload


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["mariadb", "postgres"])
async def test_pvc_to_each_database_rejects_corrupt_serialized_data(
    migration_api, corrupt_pvc_fixture, target_kind, target_factory
) -> None:
    """Corrupt rows fail validation and cannot be reported as complete."""
    source = migration_api.PVCSource(corrupt_pvc_fixture, quiesced=True)
    target = await target_factory(target_kind)

    result = await _runner(migration_api, source, target).run()

    assert result_status(result) in {"FAILED", "PARTIAL"}
    assert result.failed_datasets
    assert result.safe_to_remove_source is False


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["mariadb", "postgres"])
async def test_pvc_to_each_database_rejects_missing_column_names(
    migration_api, invalid_pvc_fixture, target_kind, target_factory
) -> None:
    """A file without schema metadata is rejected before any target write."""
    source = migration_api.PVCSource(invalid_pvc_fixture, quiesced=True)
    target = await target_factory(target_kind)

    with pytest.raises(migration_api.MigrationValidationError, match="column"):
        await _runner(migration_api, source, target).run()

    assert await target_storage(target).list_all_datasets() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["mariadb", "postgres"])
async def test_pvc_to_each_database_is_safe_only_after_verification(
    migration_api, pvc_fixture, target_kind, target_factory
) -> None:
    """The source is removable only after post-write verification succeeds."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory(target_kind)
    runner = _runner(migration_api, source, target)

    result = await runner.run()
    verification = await runner.verify()

    assert result_status(result) == "COMPLETE"
    assert verification.ok is True
    assert verification.rows_compared == 6
    assert result.safe_to_remove_source is True


@pytest.mark.asyncio
async def test_pvc_to_mariadb_and_postgres_have_identical_logical_results(
    migration_api, pvc_fixture, target_factory
) -> None:
    """Dialect choice changes transport, not the logical migrated result."""
    results = {}
    for target_kind in ("mariadb", "postgres"):
        source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
        target = await target_factory(target_kind)
        result = await _runner(migration_api, source, target).run()
        storage = target_storage(target)
        results[target_kind] = {
            name: {
                "rows": (await storage.read_data(name)).tolist(),
                "columns": await storage.get_original_column_names(name),
                "aliases": await storage.get_aliased_column_names(name),
            }
            for name in [dataset.name for dataset in pvc_fixture.datasets]
        }
        assert result_status(result) == "COMPLETE"

    assert results["mariadb"] == results["postgres"]
