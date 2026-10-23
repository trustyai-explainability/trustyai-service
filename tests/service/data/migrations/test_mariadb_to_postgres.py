"""End-to-end tests for legacy/current MariaDB to PostgreSQL migration.

The legacy source uses ``tests/resources/legacy_database_dump.sql``. The current
source is seeded in the same required CI database, not an optional second URL.
"""

from __future__ import annotations

import uuid

import numpy as np
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
async def maria_source(migration_api):
    """Create a MariaDB source from an explicitly configured test URL."""
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(live_db_url("TRUSTYAI_MIGRATION_TEST_MARIADB_URL"))
    source = migration_api.MariaDBSource(
        engine,
        schema="legacy",
        quiesced=True,
    )
    yield source
    await engine.dispose()


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
async def test_legacy_mariadb_to_postgres_preserves_all_model_datasets(
    migration_api, maria_source, postgres_target
) -> None:
    """Legacy Java-serialized MariaDB data becomes logical datasets."""
    result = await _runner(
        migration_api,
        maria_source,
        postgres_target,
        batch_size=37,
    ).run()

    assert result_status(result) == "COMPLETE"
    storage = target_storage(postgres_target)
    datasets = await storage.list_all_datasets()
    for model_id in ("model1", "model2", "model3", "model4"):
        for suffix in ("inputs", "outputs", "metadata"):
            assert f"{model_id}_{suffix}" in datasets


@pytest.mark.asyncio
async def test_legacy_mariadb_to_postgres_preserves_values_and_types(
    migration_api, maria_source, postgres_target
) -> None:
    """Representative numeric, text, boolean, and metadata values survive."""
    await _runner(migration_api, maria_source, postgres_target).run()
    storage = target_storage(postgres_target)

    model1_inputs = await storage.read_data("model1_inputs", 0, 100)
    np.testing.assert_array_equal(model1_inputs, np.array([[0, 1, 2, 3, 4]] * 100))

    model4_inputs = await storage.read_data("model4_inputs", 0, 5)
    assert model4_inputs[0].tolist() == [0.0, "i'm text-0", True, 0]
    assert model4_inputs[4].tolist() == [4.0, "i'm text-4", True, 8]

    metadata = await storage.read_data("model1_metadata", 0, 1)
    assert metadata[0][1] == "6decea54-91eb-4726-b6f8-a2f1dee8e81b"


@pytest.mark.asyncio
async def test_legacy_mariadb_to_postgres_preserves_column_aliases(
    migration_api, maria_source, postgres_target
) -> None:
    """Legacy schema mappings are translated to target aliases."""
    await _runner(migration_api, maria_source, postgres_target).run()
    aliases = await target_storage(postgres_target).get_aliased_column_names(
        "model3_inputs"
    )
    assert aliases == ["year mapped", "make mapped", "color mapped"]


@pytest.mark.asyncio
async def test_legacy_mariadb_to_postgres_is_idempotent(
    migration_api, maria_source, postgres_target
) -> None:
    """Rerunning a completed database migration does not duplicate rows."""
    first = await _runner(migration_api, maria_source, postgres_target).run()
    second = await _runner(migration_api, maria_source, postgres_target).run()

    assert result_status(first) == "COMPLETE"
    assert result_status(second) == "COMPLETE"
    assert second.rows_written == 0
    assert await target_storage(postgres_target).dataset_rows("model1_inputs") == 100


@pytest.mark.asyncio
async def test_legacy_mariadb_to_postgres_verifies_source_and_target_counts(
    migration_api, maria_source, postgres_target
) -> None:
    """The result records comparable source and target counts."""
    runner = _runner(migration_api, maria_source, postgres_target)
    result = await runner.run()
    verification = await runner.verify()

    assert result_status(result) == "COMPLETE"
    assert verification.ok is True
    assert verification.datasets_compared >= 12
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
