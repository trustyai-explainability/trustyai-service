"""Legacy-contract and live current MariaDB to PostgreSQL migration tests.

Ordered legacy tests use an explicit-position SQLite fixture. The real legacy
MariaDB dump is tested for safe rejection, not treated as an ordered schema.
The current source uses the required live MariaDB service.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from .conftest import (
    assert_dataset_matches,
    live_db_url,
    reset_target,
    result_status,
    target_storage,
)
from .test_migration_contract import _runner


@pytest_asyncio.fixture
async def ordered_legacy_source(migration_api, ordered_legacy_engine):
    """Read the explicit-position legacy contract, not an unordered dump."""
    return migration_api.MariaDBSource(
        ordered_legacy_engine,
        schema="legacy",
        quiesced=True,
    )


@pytest_asyncio.fixture
async def postgres_target(migration_api):
    """Create an isolated PostgreSQL target."""
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(live_db_url("TRUSTYAI_MIGRATION_TEST_POSTGRES_URL"))
    target = migration_api.DatabaseTarget(
        engine,
        migration_namespace=f"maria_to_postgres_{uuid.uuid4().hex}",
    )
    await reset_target(target)
    yield target
    await reset_target(target)
    await engine.dispose()


@pytest.mark.asyncio
async def test_ordered_legacy_to_postgres_preserves_all_model_datasets(
    migration_api, ordered_legacy_source, postgres_target
) -> None:
    """Legacy Java-serialized MariaDB data becomes logical datasets."""
    result = await _runner(
        migration_api,
        ordered_legacy_source,
        postgres_target,
        batch_size=37,
    ).run()

    assert result_status(result) == "COMPLETE"
    storage = target_storage(postgres_target)
    datasets = await storage.list_all_datasets()
    assert set(datasets) == {"model_inputs", "model_outputs", "model_metadata"}


@pytest.mark.asyncio
async def test_ordered_legacy_to_postgres_preserves_values_and_row_metadata(
    migration_api, ordered_legacy_source, postgres_target
) -> None:
    """Explicit cell positions preserve feature values and internal metadata."""
    await _runner(migration_api, ordered_legacy_source, postgres_target).run()
    storage = target_storage(postgres_target)

    assert (await storage.read_data("model_inputs")).tolist() == [
        ["a-1", "b-1"],
        ["a-5", "b-5"],
    ]
    assert (await storage.read_data("model_outputs")).tolist() == [["out-1"], ["out-5"]]
    assert (await storage.read_data("model_metadata", 0, 1)).tolist() == [
        ["time", "row-1", "tag"]
    ]


@pytest.mark.asyncio
async def test_ordered_legacy_to_postgres_preserves_column_aliases(
    migration_api, ordered_legacy_source, postgres_target
) -> None:
    """Legacy schema mappings are translated to target aliases."""
    await _runner(migration_api, ordered_legacy_source, postgres_target).run()
    aliases = await target_storage(postgres_target).get_aliased_column_names(
        "model_inputs"
    )
    assert aliases == ["Feature A", "feature_b"]


@pytest.mark.asyncio
async def test_ordered_legacy_to_postgres_is_idempotent(
    migration_api, ordered_legacy_source, postgres_target
) -> None:
    """Rerunning a completed database migration does not duplicate rows."""
    first = await _runner(migration_api, ordered_legacy_source, postgres_target).run()
    second = await _runner(migration_api, ordered_legacy_source, postgres_target).run()

    assert result_status(first) == "COMPLETE"
    assert result_status(second) == "COMPLETE"
    assert second.rows_written == 0
    assert await target_storage(postgres_target).dataset_rows("model_inputs") == 2


@pytest.mark.asyncio
async def test_ordered_legacy_to_postgres_verifies_source_and_target_counts(
    migration_api, ordered_legacy_source, postgres_target
) -> None:
    """The result records comparable source and target counts."""
    runner = _runner(migration_api, ordered_legacy_source, postgres_target)
    result = await runner.run()
    verification = await runner.verify()

    assert result_status(result) == "COMPLETE"
    assert verification.ok is True
    assert verification.datasets_compared == 3
    assert verification.source_rows == verification.target_rows
    assert result.safe_to_remove_source is True


@pytest.mark.asyncio
async def test_current_v2_mariadb_to_postgres_path_is_explicit(
    migration_api, postgres_target, pvc_fixture, target_factory
) -> None:
    """The required live database gate covers the current schema too."""
    maria_target = await target_factory("mariadb")
    seeded = await _runner(
        migration_api,
        migration_api.PVCSource(pvc_fixture.directory, quiesced=True),
        maria_target,
    ).run()
    assert result_status(seeded) == "COMPLETE"
    source = migration_api.MariaDBSource(
        maria_target.engine,
        schema="trustyai_v2",
        quiesced=True,
    )
    result = await _runner(migration_api, source, postgres_target).run()

    assert result_status(result) == "COMPLETE"
    for dataset in pvc_fixture.datasets:
        await assert_dataset_matches(postgres_target, dataset)


@pytest.mark.asyncio
async def test_unordered_legacy_mariadb_is_rejected_before_postgres_writes(
    migration_api, postgres_target
):
    """The real unordered fixture must not be passed off as a safe legacy import."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(live_db_url("TRUSTYAI_MIGRATION_TEST_MARIADB_URL"))
    try:
        source = migration_api.MariaDBSource(engine, schema="legacy", quiesced=True)
        with pytest.raises(
            migration_api.MigrationValidationError, match="explicit column positions"
        ):
            await _runner(migration_api, source, postgres_target).run()
        assert await target_storage(postgres_target).list_all_datasets() == []
    finally:
        await engine.dispose()
