"""End-to-end tests for legacy/current MariaDB to PostgreSQL migration.

The MariaDB source is intentionally tested through a real seeded database when
the integration URLs are supplied. The fixture is compatible with the
existing ``tests/resources/legacy_database_dump.sql`` dataset and also leaves
room for a current ``trustyai_v2_*`` fixture after the PostgreSQL PR is
rebased.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
import pytest_asyncio

from .conftest import live_db_url, result_status, target_storage
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
    await target.reset_for_test()
    yield target
    await target.reset_for_test()
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
    migration_api, postgres_target
) -> None:
    """The suite reserves a separate test for the current trustyai_v2 schema."""
    # No seeded v2 fixture exists yet, so this URL stays optional everywhere.
    url = live_db_url("TRUSTYAI_MIGRATION_TEST_MARIADB_V2_URL", required=False)

    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import create_async_engine

    source_engine = create_async_engine(url)
    source = migration_api.MariaDBSource(
        source_engine,
        schema="trustyai_v2",
        quiesced=True,
    )
    try:
        result = await _runner(migration_api, source, postgres_target).run()
    finally:
        await source_engine.dispose()

    assert result_status(result) == "COMPLETE"
