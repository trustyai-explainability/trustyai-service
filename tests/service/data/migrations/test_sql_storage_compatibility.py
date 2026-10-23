"""Compatibility tests for the canonical v2 SQL storage contract."""

from __future__ import annotations

import json
import ssl
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest

from trustyai_service.service.data.migrations.contracts import (
    DatasetDescriptor,
    PartialPayloadRecord,
)
from trustyai_service.service.serialization.detection import safe_gzip_decompress


def _descriptor(
    *,
    name: str = "model_inputs",
    shape: tuple[int, ...] = (2,),
    rows: int = 0,
) -> DatasetDescriptor:
    """Build a descriptor with the canonical SQL storage shape."""
    return DatasetDescriptor(
        name=name,
        column_names=("left", "right")[: shape[0]],
        aliases=("Left", "Right")[: shape[0]],
        column_types=("DOUBLE", "DOUBLE")[: shape[0]],
        row_count=rows,
        shape=shape,
    )


@pytest.mark.asyncio
async def test_database_target_writes_canonical_v2_schema(sqlite_target) -> None:
    """The target must be readable by the service's SQLStorage implementation."""
    from sqlalchemy import inspect, select, text

    target = sqlite_target
    descriptor = _descriptor(rows=2)
    await target.ensure_dataset(descriptor)
    await target.write_batch(
        descriptor,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=object),
    )

    async with target.engine.connect() as connection:
        table_names = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )
        assert "trustyai_v2_table_reference" in table_names
        assert "trustyai_v2_partial_payloads" in table_names
        assert "trustyai_v2_dataset_1" in table_names

        metadata = (
            await connection.execute(
                select(target.storage.reference_table.c.metadata).where(
                    target.storage.reference_table.c.dataset_name == descriptor.name
                )
            )
        ).scalar_one()
        assert metadata["column_names"] == ["left", "right"]
        assert metadata["aliased_names"] == ["Left", "Right"]
        assert metadata["shape"] == [-1, 2]
        raw_rows = await connection.execute(
            text(
                "SELECT column_0, column_1 FROM trustyai_v2_dataset_1 ORDER BY row_idx"
            )
        )
        decoded = [
            [
                json.loads(safe_gzip_decompress(bytes(cell)).decode("utf-8"))
                for cell in row
            ]
            for row in raw_rows
        ]
        assert decoded == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.asyncio
async def test_database_target_preserves_multidimensional_shape(sqlite_target) -> None:
    """The canonical metadata must retain dimensions beyond the column axis."""
    descriptor = _descriptor(name="tensor_inputs", shape=(2, 2), rows=2)
    rows = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=object,
    )

    await sqlite_target.ensure_dataset(descriptor)
    await sqlite_target.write_batch(descriptor, rows)

    assert await sqlite_target.storage.dataset_shape(descriptor.name) == (2, 2, 2)
    np.testing.assert_equal(
        await sqlite_target.storage.read_data(descriptor.name), rows
    )


@pytest.mark.asyncio
async def test_database_target_creates_empty_canonical_dataset(sqlite_target) -> None:
    """Empty source datasets must still be visible to the upgraded service."""
    descriptor = _descriptor(name="empty_inputs", rows=0)

    await sqlite_target.ensure_dataset(descriptor)

    assert await sqlite_target.storage.dataset_exists(descriptor.name)
    assert await sqlite_target.storage.dataset_rows(descriptor.name) == 0
    assert await sqlite_target.storage.dataset_shape(descriptor.name) == (0, 2)
    empty_rows = await sqlite_target.storage.read_data(descriptor.name)
    assert empty_rows.shape == (0, 2)


@pytest.mark.asyncio
async def test_database_target_repairs_missing_existing_dataset_table(
    sqlite_target,
) -> None:
    """An existing reference row with missing DDL is repaired before appending."""
    from sqlalchemy import inspect, select, text

    storage = sqlite_target.storage
    await storage.ensure_dataset("repair_inputs", ["left", "right"])
    async with storage.engine.connect() as connection:
        table_idx = (
            await connection.execute(
                select(storage.reference_table.c.table_idx).where(
                    storage.reference_table.c.dataset_name == "repair_inputs"
                )
            )
        ).scalar_one()

    async with storage.engine.begin() as connection:
        await connection.execute(text(f"DROP TABLE trustyai_v2_dataset_{table_idx}"))

    await storage.ensure_dataset("repair_inputs", ["left", "right"])

    async with storage.engine.connect() as connection:
        tables = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )
    assert f"trustyai_v2_dataset_{table_idx}" in tables

    await storage.write_data(
        "repair_inputs",
        np.asarray([[1.0, 2.0]], dtype=object),
        ["left", "right"],
    )
    assert await storage.dataset_rows("repair_inputs") == 1


@pytest.mark.asyncio
async def test_database_target_uses_canonical_partial_payload_columns(
    sqlite_target, partial_payload
) -> None:
    """Partial payloads must use the schema consumed by SQLStorage."""
    from sqlalchemy import inspect

    await sqlite_target.persist_partial(
        PartialPayloadRecord(
            payload_id="pending-1",
            is_input=True,
            is_modelmesh=False,
            payload=partial_payload,
        )
    )

    async with sqlite_target.engine.connect() as connection:
        columns = await connection.run_sync(
            lambda sync_connection: [
                column["name"]
                for column in inspect(sync_connection).get_columns(
                    "trustyai_v2_partial_payloads"
                )
            ]
        )
    assert columns == ["payload_id", "is_input", "payload_data"]
    assert (
        await sqlite_target.storage.get_partial_payload(
            "pending-1", is_input=True, is_modelmesh=False
        )
        == partial_payload
    )


def test_cli_builds_target_from_operator_database_environment(monkeypatch) -> None:
    """The CLI accepts the operator's DATABASE storage environment contract."""
    from trustyai_service.service.data.migrations.cli import parse_args

    credential_env_name = "QUARKUS_DATASOURCE_" + "PASS" + "WORD"
    values = {
        "SERVICE_STORAGE_FORMAT": "DATABASE",
        "QUARKUS_DATASOURCE_DB_KIND": "postgresql",
        "QUARKUS_DATASOURCE_USERNAME": "trusty",
        credential_env_name: "db-test-value",
        "DATABASE_SERVICE": "postgres.example",
        "DATABASE_PORT": "5433",
        "DATABASE_NAME": "trustyai",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TRUSTYAI_MIGRATION_TARGET_URL", raising=False)

    from trustyai_service.service.data.migrations.cli import target_url_from_environment

    args = parse_args(
        [
            "--source",
            "pvc",
            "--source-path",
            "source",
            "--quiesced",
        ]
    )

    target_url = target_url_from_environment()
    parsed_url = urlsplit(target_url)
    assert args.target_url == target_url
    assert parsed_url.scheme == "postgresql+asyncpg"
    assert parsed_url.username == "trusty"
    assert parsed_url.password == values[credential_env_name]
    assert parsed_url.hostname == "postgres.example"
    assert parsed_url.port == 5433
    assert parsed_url.path == "/trustyai"


def test_cli_maps_operator_tls_and_pool_tuning(monkeypatch) -> None:
    """Operator ConfigMap tuning and Secret-mounted TLS reach the migration engine."""
    from trustyai_service.service.data.migrations.cli import (
        engine_options_from_environment,
    )

    ca_file = Path(ssl.get_default_verify_paths().cafile or "")
    if not ca_file.is_file():
        pytest.skip("the test environment has no public CA bundle")
    values = {
        "DATABASE_TLS_CA_CERT": str(ca_file),
        "DATABASE_POOL_SIZE": "3",
        "DATABASE_MAX_OVERFLOW": "4",
        "DATABASE_POOL_TIMEOUT": "11",
        "DATABASE_POOL_RECYCLE": "22",
        "DATABASE_APPLICATION_NAME": "migration-job",
        "DATABASE_STATEMENT_TIMEOUT_MS": "5000",
        "DATABASE_SYNCHRONOUS_COMMIT": "off",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    options = engine_options_from_environment("postgresql+asyncpg://db/trustyai")

    assert options["pool_size"] == 3
    assert options["max_overflow"] == 4
    assert options["pool_timeout"] == 11
    assert options["pool_recycle"] == 22
    connect_args = options["connect_args"]
    assert isinstance(connect_args, dict)
    assert isinstance(connect_args["ssl"], ssl.SSLContext)
    assert connect_args["server_settings"] == {
        "application_name": "migration-job",
        "statement_timeout": "5000",
        "synchronous_commit": "off",
    }


def test_migration_dependencies_are_explicit_and_container_visible() -> None:
    """The CLI's optional drivers must be installable in the service image."""
    pyproject = Path("pyproject.toml")
    with pyproject.open("rb") as stream:
        project = tomllib.load(stream)
    migration = project["project"]["optional-dependencies"]["migration"]

    assert any("sqlalchemy" in requirement for requirement in migration)
    assert any("asyncpg" in requirement for requirement in migration)
    assert any("asyncmy" in requirement for requirement in migration)
    assert any("javaobj-py3" in requirement for requirement in migration)
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "--extra migration" in requirements
    assert "asyncpg==" in requirements
    assert "asyncmy==" in requirements
    assert "javaobj-py3==" in requirements
    assert "sqlalchemy==" in requirements


def test_cli_rejects_configured_missing_ca_without_insecure_opt_out(
    monkeypatch, tmp_path: Path
) -> None:
    """A mounted-but-unusable CA cannot silently downgrade database TLS."""
    from trustyai_service.service.data.migrations.cli import (
        engine_options_from_environment,
    )

    monkeypatch.setenv("DATABASE_TLS_CA_CERT", str(tmp_path / "missing.crt"))
    monkeypatch.setenv("DATABASE_ALLOW_INSECURE_TLS", "false")

    with pytest.raises(ValueError, match="usable CA certificate"):
        engine_options_from_environment("postgresql+asyncpg://db/trustyai")
