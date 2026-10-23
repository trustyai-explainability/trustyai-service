"""Fixtures shared by storage migration contract and integration tests.

Optional migration adapters are loaded lazily so dependency errors remain
localized to migration tests.

The public API specified by these tests is:

* ``trustyai_service.service.data.migrations.PVCSource``
* ``trustyai_service.service.data.migrations.PVCCSVSource``
* ``trustyai_service.service.data.migrations.MariaDBSource``
* ``trustyai_service.service.data.migrations.DatabaseTarget``
* ``trustyai_service.service.data.migrations.MigrationConfig``
* ``trustyai_service.service.data.migrations.MigrationRunner``
* ``trustyai_service.service.data.migrations.MigrationStatus``

Sources expose logical datasets and partial payloads. Targets expose the
resulting ``StorageInterface`` through ``storage`` for black-box verification.
The runner owns planning, tracking, retries, and verification.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import pytest
import pytest_asyncio

from trustyai_service.endpoints.consumer import KServeData, KServeInferenceRequest
from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.data.storage.pvc import (
    MAX_VOID_TYPE_LENGTH,
    PARTIAL_INPUT_NAME,
)
from trustyai_service.service.serialization import serialize_model, serialize_rows


@dataclass(frozen=True)
class ExpectedDataset:
    """Logical dataset expected after migration."""

    name: str
    rows: tuple[tuple[Any, ...], ...]
    column_names: tuple[str, ...]
    aliases: tuple[str, ...]
    column_types: tuple[str, ...]


@dataclass(frozen=True)
class PVCFixture:
    """PVC fixture directory and its logical contents."""

    directory: Path
    datasets: tuple[ExpectedDataset, ...]
    file_hashes: dict[str, str]


def _set_dataset_attributes(
    dataset: h5py.Dataset,
    *,
    column_names: tuple[str, ...],
    aliases: tuple[str, ...],
    column_types: tuple[str, ...],
    is_bytes: bool,
) -> None:
    """Set the attributes used by the PVC source adapter."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    dataset.attrs["column_names"] = np.asarray(column_names, dtype=string_dtype)
    dataset.attrs["column_aliases"] = np.asarray(aliases, dtype=string_dtype)
    dataset.attrs["column_types"] = np.asarray(column_types, dtype=string_dtype)
    dataset.attrs["is_bytes"] = is_bytes


def _hash_files(directory: Path) -> dict[str, str]:
    """Return SHA-256 hashes for every source file."""
    hashes: dict[str, str] = {}
    for path in sorted(directory.glob("*.hdf5")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _load_migration_module() -> ModuleType:
    """Load the migration public API, with an actionable TDD failure."""
    try:
        module = importlib.import_module("trustyai_service.service.data.migrations")
    except ModuleNotFoundError as error:
        pytest.fail(
            "The migration framework is not present yet. Add "
            "trustyai_service.service.data.migrations before enabling this suite: "
            f"{error}",
            pytrace=False,
        )

    required = (
        "PVCSource",
        "PVCCSVSource",
        "MariaDBSource",
        "DatabaseTarget",
        "MigrationConfig",
        "MigrationRunner",
        "MigrationStatus",
        "MigrationPreflightError",
        "MigrationValidationError",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        pytest.fail(
            "The migration public API is incomplete; missing: " + ", ".join(missing),
            pytrace=False,
        )
    return module


@pytest.fixture
def migration_api() -> ModuleType:
    """Return the migration public API under test."""
    return _load_migration_module()


@pytest.fixture
def pvc_fixture(tmp_path: Path) -> PVCFixture:
    """Create representative PVC files, including encoded and empty data."""
    prefix = "migration_" + tmp_path.name.replace("-", "_")

    numeric_name = f"{prefix}_inputs"
    numeric_rows = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    numeric_path = tmp_path / f"{numeric_name}_trustyai.hdf5"
    with h5py.File(numeric_path, "w") as h5_file:
        dataset = h5_file.create_dataset(
            numeric_name,
            data=np.asarray(numeric_rows, dtype=np.float64),
        )
        _set_dataset_attributes(
            dataset,
            column_names=("age", "income", "score"),
            aliases=("Age", "Annual income", "Score"),
            column_types=("DOUBLE", "DOUBLE", "DOUBLE"),
            is_bytes=False,
        )

    encoded_name = f"{prefix}_outputs"
    encoded_rows = (
        ("approved", True, b"\x00\x01", {"rank": 1}),
        ("review", False, b"\x02\x03", {"rank": 2}),
        ("rejected", False, b"\x04\x05", {"rank": 3}),
    )
    encoded_path = tmp_path / f"{encoded_name}_trustyai.hdf5"
    serialized = serialize_rows(list(encoded_rows), MAX_VOID_TYPE_LENGTH)
    with h5py.File(encoded_path, "w") as h5_file:
        dataset = h5_file.create_dataset(encoded_name, data=serialized)
        _set_dataset_attributes(
            dataset,
            column_names=("decision", "eligible", "raw", "details"),
            aliases=("Decision", "Eligible", "Raw bytes", "Details"),
            column_types=("STRING", "BOOLEAN", "BYTES", "JSON"),
            is_bytes=True,
        )

    empty_name = f"{prefix}_empty"
    empty_path = tmp_path / f"{empty_name}_trustyai.hdf5"
    with h5py.File(empty_path, "w") as h5_file:
        dataset = h5_file.create_dataset(
            empty_name,
            shape=(0, 2),
            dtype=np.float64,
        )
        _set_dataset_attributes(
            dataset,
            column_names=("left", "right"),
            aliases=("left", "right"),
            column_types=("DOUBLE", "DOUBLE"),
            is_bytes=False,
        )

    datasets = (
        ExpectedDataset(
            name=numeric_name,
            rows=numeric_rows,
            column_names=("age", "income", "score"),
            aliases=("Age", "Annual income", "Score"),
            column_types=("DOUBLE", "DOUBLE", "DOUBLE"),
        ),
        ExpectedDataset(
            name=encoded_name,
            rows=encoded_rows,
            column_names=("decision", "eligible", "raw", "details"),
            aliases=("Decision", "Eligible", "Raw bytes", "Details"),
            column_types=("STRING", "BOOLEAN", "BYTES", "JSON"),
        ),
        ExpectedDataset(
            name=empty_name,
            rows=(),
            column_names=("left", "right"),
            aliases=("left", "right"),
            column_types=("DOUBLE", "DOUBLE"),
        ),
    )
    return PVCFixture(
        directory=tmp_path,
        datasets=datasets,
        file_hashes=_hash_files(tmp_path),
    )


def assert_rows_equal(
    actual: np.ndarray, expected: tuple[tuple[Any, ...], ...]
) -> None:
    """Compare logical rows without losing bytes or nested JSON values."""
    actual_rows = [tuple(row.tolist()) for row in actual]
    assert actual_rows == [tuple(row) for row in expected]


def target_storage(target: Any) -> Any:
    """Return the target's black-box storage view."""
    return getattr(target, "storage", target)


async def reset_target(target):
    """Reset only an explicitly isolated test database and its checkpoints."""
    await target.storage.reset_for_test()
    await target.checkpoint_storage.reset_for_test()


def create_test_target(api, engine, *, fail_after_batches=None, **kwargs):
    """Keep fault injection and destructive test setup out of production APIs."""

    class InstrumentedTarget(api.DatabaseTarget):
        def __init__(self):
            super().__init__(engine, **kwargs)
            self.remaining = fail_after_batches

        async def write_batch(self, dataset, rows):
            if self.remaining is not None:
                if self.remaining == 0:
                    msg = "injected target batch failure"
                    raise api.MigrationWriteError(msg)
                self.remaining -= 1
            await super().write_batch(dataset, rows)

        async def reset_for_test(self):
            await reset_target(self)

        def clear_failure(self):
            self.remaining = None

    return InstrumentedTarget()


async def assert_dataset_matches(target: Any, expected: ExpectedDataset) -> None:
    """Verify data and metadata through the storage contract only."""
    storage = target_storage(target)
    assert await storage.dataset_exists(expected.name)
    assert await storage.dataset_rows(expected.name) == len(expected.rows)
    assert await storage.get_original_column_names(expected.name) == list(
        expected.column_names
    )
    assert await storage.get_aliased_column_names(expected.name) == list(
        expected.aliases
    )
    assert await storage.get_column_types(expected.name) == list(expected.column_types)
    assert_rows_equal(await storage.read_data(expected.name), expected.rows)


def result_status(result: Any) -> str:
    """Normalize enum and string result statuses for assertions."""
    status = result.status
    return getattr(status, "value", status)


def result_json(result: Any) -> dict[str, Any]:
    """Return a stable result representation for idempotence assertions."""
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if hasattr(result, "__dict__"):
        return json.loads(json.dumps(result.__dict__, default=str))
    return {"status": result_status(result)}


def live_db_url(env_name: str, *, required: bool = True) -> str:
    """Return a configured live-database URL for migration integration tests.

    Local runs without a database skip. CI sets
    ``TRUSTYAI_REQUIRE_LIVE_SQL_TESTS`` so a missing URL fails instead of
    silently skipping coverage. Sources without a seeded fixture (the v2
    MariaDB schema) pass ``required=False`` to stay optional everywhere.
    """
    import os

    url = os.environ.get(env_name)
    if url:
        return url
    reason = f"set {env_name} to run live-database migration tests"
    if required and os.environ.get("TRUSTYAI_REQUIRE_LIVE_SQL_TESTS") == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


@pytest_asyncio.fixture
async def target_factory(migration_api):
    """Build a disposable SQL target from an explicit test URL."""
    import uuid

    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import create_async_engine

    targets = []

    async def factory(kind: str):
        url = live_db_url(
            "TRUSTYAI_MIGRATION_TEST_MARIADB_URL"
            if kind == "mariadb"
            else "TRUSTYAI_MIGRATION_TEST_POSTGRES_URL"
        )
        engine = create_async_engine(url)
        target = create_test_target(
            migration_api,
            engine,
            migration_namespace=f"test_{uuid.uuid4().hex}",
        )
        await target.reset_for_test()
        targets.append(target)
        return target

    yield factory
    for target in targets:
        try:
            await target.reset_for_test()
        finally:
            await target.engine.dispose()


@pytest_asyncio.fixture
async def ordered_legacy_engine():
    """Legacy contract with explicit cell positions, not inferred insertion order."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from .test_java_values import serialized_legacy_string

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    statements = [
        "CREATE TABLE DataframeMetadata (id TEXT PRIMARY KEY)",
        "CREATE TABLE DataframeMetadata_names (DataframeMetadata_id TEXT, names TEXT, names_ORDER INTEGER)",
        "CREATE TABLE DataframeMetadata_inputs (DataframeMetadata_id TEXT, inputs INTEGER, inputs_ORDER INTEGER)",
        "CREATE TABLE DataframeRow (dbId INTEGER PRIMARY KEY, timestamp TEXT, modelId TEXT, rowId TEXT, tag TEXT)",
        "CREATE TABLE DataframeRow_Values (DataframeRow_dbId INTEGER, order_column INTEGER, serializableObject BLOB)",
        "CREATE TABLE StorageMetadata_StorageSchema (StorageMetadata_modelId TEXT, schemas_id INTEGER)",
        "CREATE TABLE storageSchema_mappedItems (Schema_id INTEGER, mappedItems_id INTEGER, mappedItems_KEY TEXT)",
        "CREATE TABLE storageSchema_originalItems (items_id INTEGER, items_KEY TEXT)",
        "INSERT INTO DataframeMetadata VALUES ('model')",
        "INSERT INTO DataframeMetadata_names VALUES ('model', 'feature_b', 2), ('model', 'feature_a', 0), ('model', 'result', 1)",
        "INSERT INTO DataframeMetadata_inputs VALUES ('model', 0, 1), ('model', 1, 2), ('model', 1, 0)",
        "INSERT INTO StorageMetadata_StorageSchema VALUES ('model', 1)",
        "INSERT INTO storageSchema_mappedItems VALUES (1, 1, 'Feature A')",
        "INSERT INTO storageSchema_originalItems VALUES (1, 'feature_a')",
    ]
    try:
        async with engine.begin() as connection:
            for statement in statements:
                await connection.execute(text(statement))
            for row_id in [1, 5]:
                await connection.execute(
                    text(
                        "INSERT INTO DataframeRow VALUES (:id, 'time', 'model', :row_id, 'tag')"
                    ),
                    {"id": row_id, "row_id": f"row-{row_id}"},
                )
                for column, value in enumerate(
                    [f"a-{row_id}", f"out-{row_id}", f"b-{row_id}"]
                ):
                    await connection.execute(
                        text(
                            "INSERT INTO DataframeRow_Values VALUES (:id, :column, :value)"
                        ),
                        {
                            "id": row_id,
                            "column": column,
                            "value": serialized_legacy_string(value),
                        },
                    )
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def sqlite_target(migration_api):
    """Provide an isolated in-memory target for fast framework tests."""
    pytest.importorskip("aiosqlite")
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    target = create_test_target(
        migration_api,
        engine,
        migration_namespace="sqlite_unit",
    )
    await target.reset_for_test()
    yield target
    await engine.dispose()


@pytest_asyncio.fixture
async def failing_target_factory(migration_api):
    """Build a target that injects one deterministic write failure."""
    import uuid

    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import create_async_engine

    targets = []

    async def factory(kind: str, *, fail_after_batches: int):
        url = live_db_url(
            "TRUSTYAI_MIGRATION_TEST_MARIADB_URL"
            if kind == "mariadb"
            else "TRUSTYAI_MIGRATION_TEST_POSTGRES_URL"
        )
        engine = create_async_engine(url)
        target = create_test_target(
            migration_api,
            engine,
            migration_namespace=f"failure_{kind}_{uuid.uuid4().hex}",
            fail_after_batches=fail_after_batches,
        )
        await target.reset_for_test()
        targets.append(target)
        return target

    yield factory
    for target in targets:
        try:
            await target.reset_for_test()
        finally:
            await target.engine.dispose()


@pytest.fixture
def partial_payload() -> KServeInferenceRequest:
    """Return a representative KServe partial payload."""
    return KServeInferenceRequest(
        id="pending-1",
        inputs=[
            KServeData(
                name="features",
                shape=[1, 2],
                datatype="FP64",
                data=[1.5, 2.5],
            )
        ],
    )


@pytest.fixture
def modelmesh_partial_payload() -> PartialPayload:
    """Return a representative ModelMesh partial payload."""
    return PartialPayload(data="cHJvdG9idWY=", metadata={"model_id": "mm-model"})


@pytest.fixture
def pvc_fixture_with_partial_payload(
    pvc_fixture: PVCFixture,
    partial_payload: KServeInferenceRequest,
) -> PVCFixture:
    """Add a persisted partial input payload to the representative PVC."""
    partial_path = pvc_fixture.directory / f"{PARTIAL_INPUT_NAME}_trustyai.hdf5"
    with h5py.File(partial_path, "w") as h5_file:
        dataset = h5_file.create_dataset(
            PARTIAL_INPUT_NAME, dtype="f", track_order=True
        )
        dataset.attrs["pending-1"] = np.void(serialize_model(partial_payload))
    return pvc_fixture


@pytest.fixture
def pvc_fixture_with_modelmesh_partial_payload(
    pvc_fixture_with_partial_payload: PVCFixture,
    modelmesh_partial_payload: PartialPayload,
) -> PVCFixture:
    """Add a persisted ModelMesh partial input payload to the PVC."""
    partial_path = pvc_fixture_with_partial_payload.directory / (
        f"{PARTIAL_INPUT_NAME}_trustyai.hdf5"
    )
    with h5py.File(partial_path, "a") as h5_file:
        h5_file[PARTIAL_INPUT_NAME].attrs["mm-1"] = np.void(
            serialize_model(modelmesh_partial_payload)
        )
    return pvc_fixture_with_partial_payload


@pytest.fixture
def corrupt_pvc_fixture(tmp_path: Path) -> Path:
    """Create an HDF5 source containing invalid compressed row bytes."""
    path = tmp_path / "corrupt_outputs_trustyai.hdf5"
    with h5py.File(path, "w") as h5_file:
        dataset = h5_file.create_dataset(
            "corrupt_outputs",
            data=np.asarray([np.void(b"not-gzip")], dtype="V8"),
        )
        dataset.attrs["column_names"] = np.asarray(["value"], dtype=h5py.string_dtype())
        dataset.attrs["is_bytes"] = True
    return tmp_path


@pytest.fixture
def invalid_pvc_fixture(tmp_path: Path) -> Path:
    """Create an HDF5 source missing required column metadata."""
    path = tmp_path / "invalid_inputs_trustyai.hdf5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("invalid_inputs", data=np.ones((1, 1)))
    return tmp_path
