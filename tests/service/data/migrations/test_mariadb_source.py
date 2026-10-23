"""Unit coverage for the current MariaDB logical source adapter."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest
import pytest_asyncio

from trustyai_service.service.serialization.encoders import json_encoder

from .conftest import live_db_url, result_status, target_storage
from .test_migration_contract import _runner


@pytest.mark.asyncio
async def test_legacy_reader_pages_complete_rows_without_binary_aggregation(
    migration_api,
):
    """Large cells are intact and sequential reads never use OFFSET."""
    from sqlalchemy import event, text
    from sqlalchemy.ext.asyncio import create_async_engine

    from trustyai_service.service.data.migrations.mariadb import _LegacyDatasetSpec

    from .test_java_values import serialized_legacy_string

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    statements = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def record(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TABLE DataframeRow (dbId INTEGER PRIMARY KEY, timestamp TEXT, modelId TEXT, rowId TEXT, tag TEXT)"
                )
            )
            await connection.execute(
                text(
                    "CREATE TABLE DataframeRow_Values (DataframeRow_dbId INTEGER, order_column INTEGER, serializableObject BLOB)"
                )
            )
            for row_id in [1, 5, 9]:
                await connection.execute(
                    text(
                        "INSERT INTO DataframeRow VALUES (:id, 'time', 'model', :name, 'tag')"
                    ),
                    {"id": row_id, "name": str(row_id)},
                )
                for column in range(2):
                    await connection.execute(
                        text(
                            "INSERT INTO DataframeRow_Values VALUES (:id, :column, :data)"
                        ),
                        {
                            "id": row_id,
                            "column": column,
                            "data": serialized_legacy_string(
                                f"{row_id}:{column}:" + "x" * 2_000
                            ),
                        },
                    )
        source = migration_api.MariaDBSource(engine, schema="legacy", quiesced=True)
        source._mode = "legacy"
        descriptor = migration_api.DatasetDescriptor(
            name="model_inputs",
            column_names=("x", "y"),
            aliases=("x", "y"),
            column_types=("UNKNOWN", "UNKNOWN"),
            row_count=3,
        )
        source._legacy_specs[descriptor.name] = _LegacyDatasetSpec(
            "model", "inputs", ("x", "y"), (0, 1), ("x", "y")
        )
        statements.clear()
        for start, row_id in enumerate([1, 5, 9]):
            rows = await source.read_rows(descriptor, start, 1)
            assert rows.shape == (1, 2)
            assert rows[0, 0] == f"{row_id}:0:" + "x" * 2_000
        assert not any(
            "GROUP_CONCAT" in statement or "OFFSET" in statement
            for statement in statements
        )
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "DELETE FROM DataframeRow_Values WHERE DataframeRow_dbId=1 AND order_column=1"
                )
            )
        with pytest.raises(
            migration_api.MigrationValidationError,
            match="model 'model', row 1: incorrect column",
        ):
            await source.read_rows(descriptor, 0, 1)
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def current_schema_engine(partial_payload, modelmesh_partial_payload):
    """Build a small SQLite database with the current v2 table shape."""
    pytest.importorskip("aiosqlite")
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from trustyai_service.service.serialization import serialize_model

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    metadata = {
        "column_names": ["age", "score"],
        "aliased_names": ["Age", "Score"],
        "column_types": ["DOUBLE", "DOUBLE"],
        "trustyai_service": {
            "input_tensor_name": "instances",
            "output_tensor_name": "predictions",
            "recorded_inferences": True,
        },
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
                "CREATE TABLE trustyai_v2_partial_payloads "
                "(payload_id TEXT, is_input INTEGER, payload_data BLOB)"
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
        await connection.execute(
            text(
                "INSERT INTO trustyai_v2_partial_payloads VALUES "
                "(:payload_id, :is_input, :payload_data)"
            ),
            [
                {
                    "payload_id": "kserve-1",
                    "is_input": 1,
                    "payload_data": serialize_model(partial_payload),
                },
                {
                    "payload_id": "modelmesh-1",
                    "is_input": 1,
                    "payload_data": serialize_model(modelmesh_partial_payload),
                },
            ],
        )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def legacy_mariadb_engine():
    """Connect to the optional seeded legacy MariaDB integration service."""
    pytest.importorskip("asyncmy")
    url = live_db_url("TRUSTYAI_MIGRATION_TEST_MARIADB_URL")
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
    assert manifest.datasets[0].service_metadata == {
        "input_tensor_name": "instances",
        "output_tensor_name": "predictions",
        "recorded_inferences": True,
    }
    np.testing.assert_array_equal(rows, np.asarray([[1.0, 0.5], [2.0, 0.75]]))
    with pytest.raises(ValueError, match="non-negative"):
        await source.read_rows(manifest.datasets[0], -1, 1)


@pytest.mark.asyncio
async def test_current_source_digest_includes_service_metadata(
    migration_api, current_schema_engine
) -> None:
    """Editing tensor names without editing observations invalidates the snapshot."""
    from sqlalchemy import text

    source = migration_api.MariaDBSource(
        current_schema_engine, schema="trustyai_v2", quiesced=True
    )
    before = await source.preflight()
    async with current_schema_engine.begin() as connection:
        metadata = json.loads(
            (
                await connection.execute(
                    text("SELECT metadata FROM trustyai_v2_table_reference")
                )
            ).scalar_one()
        )
        metadata["trustyai_service"]["input_tensor_name"] = "changed"
        await connection.execute(
            text("UPDATE trustyai_v2_table_reference SET metadata=:metadata"),
            {"metadata": json.dumps(metadata)},
        )
    assert (await source.preflight()).digest != before.digest


@pytest.mark.asyncio
async def test_current_mariadb_source_uses_keyset_pagination_for_sequential_reads(
    migration_api, current_schema_engine
) -> None:
    """Current-schema reads seek by row_idx without repeated OFFSET scans."""
    from sqlalchemy import event, text

    async with current_schema_engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO trustyai_v2_dataset_1 "
                "(row_idx, column_0, column_1) VALUES "
                "(:row_idx, :column_0, :column_1)"
            ),
            {
                "row_idx": 4,
                "column_0": gzip.compress(b"3.0"),
                "column_1": gzip.compress(b"0.9"),
            },
        )
        await connection.execute(
            text(
                "UPDATE trustyai_v2_table_reference SET n_rows=3 "
                "WHERE dataset_name='model_inputs'"
            )
        )

    statements: list[str] = []

    def capture_statement(*args) -> None:
        statement = args[2]
        if (
            "trustyai_v2_dataset_1" in statement
            and statement.lstrip().upper().startswith("SELECT")
        ):
            statements.append(statement)

    event.listen(
        current_schema_engine.sync_engine,
        "before_cursor_execute",
        capture_statement,
    )
    try:
        source = migration_api.MariaDBSource(
            current_schema_engine,
            schema="trustyai_v2",
            quiesced=True,
        )
        manifest = await source.preflight()
        dataset = manifest.datasets[0]
        first = await source.read_rows(dataset, 0, 1)
        second = await source.read_rows(dataset, 1, 1)
        third = await source.read_rows(dataset, 2, 1)
    finally:
        event.remove(
            current_schema_engine.sync_engine,
            "before_cursor_execute",
            capture_statement,
        )

    np.testing.assert_array_equal(first, np.asarray([[1.0, 0.5]]))
    np.testing.assert_array_equal(second, np.asarray([[2.0, 0.75]]))
    np.testing.assert_array_equal(third, np.asarray([[3.0, 0.9]]))
    assert statements
    assert all(" OFFSET " not in statement.upper() for statement in statements)
    keyset_statements = [
        statement for statement in statements if "WHERE row_idx >" in statement
    ]
    assert len(keyset_statements) == 5


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
async def test_current_mariadb_source_classifies_modelmesh_partials(
    migration_api, current_schema_engine, modelmesh_partial_payload
) -> None:
    """Stored ModelMesh partial payloads are detected instead of failing."""
    source = migration_api.MariaDBSource(
        current_schema_engine,
        schema="trustyai_v2",
        quiesced=True,
    )

    manifest = await source.preflight()

    by_id = {record.payload_id: record for record in manifest.partial_payloads}
    assert by_id["kserve-1"].is_modelmesh is False
    assert by_id["modelmesh-1"].is_modelmesh is True
    assert by_id["modelmesh-1"].payload == modelmesh_partial_payload


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
