"""Unit coverage for the current MariaDB logical source adapter."""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pytest
import pytest_asyncio

from trustyai_service.service.serialization.encoders import json_encoder

from .conftest import result_status, target_storage
from .test_migration_contract import _runner


@pytest_asyncio.fixture
async def current_schema_engine():
    """Build a small SQLite database with the current v2 table shape."""
    pytest.importorskip("aiosqlite")
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata = {
        "column_names": ["age", "score"],
        "aliased_names": ["Age", "Score"],
        "column_types": ["DOUBLE", "DOUBLE"],
    }
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE trustyai_v2_table_reference "
                "(table_idx INTEGER, dataset_name TEXT, metadata TEXT, n_rows INTEGER)"
            )
        )
        await connection.execute(
            text(
                "CREATE TABLE trustyai_v2_dataset_1 "
                "(row_idx INTEGER, column_0 BLOB, column_1 BLOB)"
            )
        )
        await connection.execute(
            text(
                "INSERT INTO trustyai_v2_table_reference VALUES "
                "(1, 'model_inputs', :metadata, 2)"
            ),
            {"metadata": json.dumps(metadata)},
        )
        rows = []
        for index, values in enumerate(((1.0, 0.5), (2.0, 0.75))):
            rows.append(
                {
                    "row_idx": index,
                    "column_0": gzip.compress(
                        json.dumps(values[0], default=json_encoder).encode()
                    ),
                    "column_1": gzip.compress(
                        json.dumps(values[1], default=json_encoder).encode()
                    ),
                }
            )
        await connection.execute(
            text(
                "INSERT INTO trustyai_v2_dataset_1 "
                "(row_idx, column_0, column_1) VALUES "
                "(:row_idx, :column_0, :column_1)"
            ),
            rows,
        )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def legacy_mariadb_engine():
    """Connect to the optional seeded legacy MariaDB integration service."""
    pytest.importorskip("asyncmy")
    url = os.environ.get("TRUSTYAI_MIGRATION_TEST_MARIADB_URL")
    if not url:
        pytest.skip("set TRUSTYAI_MIGRATION_TEST_MARIADB_URL for MariaDB integration")
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url)
    yield engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_current_mariadb_source_reads_canonical_rows(
    migration_api, current_schema_engine
) -> None:
    """Current v2 registry and blob rows become logical datasets."""
    source = migration_api.MariaDBSource(
        current_schema_engine,
        schema="trustyai_v2",
        quiesced=True,
    )

    manifest = await source.preflight()
    rows = await source.read_rows(manifest.datasets[0], 0, 2)

    assert manifest.total_files == 1
    assert manifest.datasets[0].name == "model_inputs"
    assert manifest.datasets[0].aliases == ("Age", "Score")
    np.testing.assert_array_equal(rows, np.asarray([[1.0, 0.5], [2.0, 0.75]]))
    with pytest.raises(ValueError, match="non-negative"):
        await source.read_rows(manifest.datasets[0], -1, 1)


@pytest.mark.asyncio
async def test_current_mariadb_source_digest_includes_row_contents(
    migration_api, current_schema_engine
) -> None:
    """A row edit must invalidate the source snapshot digest."""
    from sqlalchemy import text

    source = migration_api.MariaDBSource(
        current_schema_engine,
        schema="trustyai_v2",
        quiesced=True,
    )
    before = await source.preflight()

    async with current_schema_engine.begin() as connection:
        await connection.execute(
            text("UPDATE trustyai_v2_dataset_1 SET column_0=:column_0 WHERE row_idx=0"),
            {"column_0": gzip.compress(b"9.0")},
        )

    after = await source.preflight()

    assert after.digest != before.digest


@pytest.mark.asyncio
async def test_current_mariadb_source_rejects_stale_registry_count(
    migration_api, current_schema_engine
) -> None:
    """A stale registry count must not silently omit source rows."""
    from sqlalchemy import text

    async with current_schema_engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO trustyai_v2_dataset_1 "
                "(row_idx, column_0, column_1) VALUES "
                "(:row_idx, :column_0, :column_1)"
            ),
            {
                "row_idx": 2,
                "column_0": gzip.compress(b"3.0"),
                "column_1": gzip.compress(b"0.9"),
            },
        )
    source = migration_api.MariaDBSource(
        current_schema_engine,
        schema="trustyai_v2",
        quiesced=True,
    )

    with pytest.raises(migration_api.MigrationValidationError, match="inconsistent"):
        await source.preflight()


@pytest.mark.asyncio
async def test_current_mariadb_source_runs_through_canonical_target(
    migration_api, current_schema_engine, sqlite_target
) -> None:
    """The current v2 source adapter uses the same runner as legacy data."""
    source = migration_api.MariaDBSource(
        current_schema_engine,
        schema="trustyai_v2",
        quiesced=True,
    )

    result = await _runner(migration_api, source, sqlite_target).run()

    assert result_status(result) == "COMPLETE"
    np.testing.assert_array_equal(
        await target_storage(sqlite_target).read_data("model_inputs"),
        np.asarray([[1.0, 0.5], [2.0, 0.75]]),
    )


@pytest.mark.asyncio
async def test_auto_mariadb_source_rejects_ambiguous_schemas(
    migration_api, current_schema_engine
) -> None:
    """Auto-detection cannot silently choose stale data when both schemas exist."""
    from sqlalchemy import text

    async with current_schema_engine.begin() as connection:
        await connection.execute(
            text("CREATE TABLE DataframeMetadata (id TEXT PRIMARY KEY)")
        )

    source = migration_api.MariaDBSource(
        current_schema_engine,
        schema="auto",
        quiesced=True,
    )

    with pytest.raises(migration_api.MigrationPreflightError, match="both"):
        await source.preflight()


@pytest.mark.asyncio
async def test_legacy_mariadb_source_transfers_java_rows_to_sqlite(
    migration_api, legacy_mariadb_engine, sqlite_target
) -> None:
    """The real legacy schema streams ordered Java values through the runner."""
    source = migration_api.MariaDBSource(
        legacy_mariadb_engine,
        schema="legacy",
        quiesced=True,
    )

    manifest = await source.preflight()
    assert len(manifest.datasets) == 12
    assert manifest.datasets[0].row_count == 100

    result = await _runner(migration_api, source, sqlite_target, batch_size=37).run()

    assert result_status(result) == "COMPLETE"
    storage = target_storage(sqlite_target)
    np.testing.assert_array_equal(
        await storage.read_data("model1_inputs", 0, 3),
        np.asarray([[0, 1, 2, 3, 4]] * 3),
    )
    assert (await storage.read_data("model4_inputs", 0, 1))[0].tolist() == [
        0.0,
        "i'm text-0",
        True,
        0,
    ]
    assert await storage.get_aliased_column_names("model3_inputs") == [
        "year mapped",
        "make mapped",
        "color mapped",
    ]
