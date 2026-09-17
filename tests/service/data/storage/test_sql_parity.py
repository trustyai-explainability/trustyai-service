"""Shared behavior-parity suite across all SQL backends.

The same scenarios run against every available SQL backend and assert identical
observable behavior. This is the primary guard for the shared ``SQLStorage``
base (see ``docs/sql-storage-backends-plan.md`` §8).

- **SQLite** (``:memory:``) always runs -- fast, serverless, no external DB.
- **PostgreSQL** runs when a server is reachable at 127.0.0.1:5432
  (``podman compose -f tests/resources/compose-local-postgres.yaml up``).
- **MariaDB** runs when a server is reachable at 127.0.0.1:3306 and the
  ``mariadb`` driver is importable.
"""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("sqlalchemy")

from tests.service.data.storage.sql_test_resources import SQLTestResources
from trustyai_service.service.data.modelmesh_parser import PartialPayload

if TYPE_CHECKING:
    from collections.abc import Iterator

    from trustyai_service.service.data.storage.sql.base import SQLStorage

ALPHABET = "abcdefghijklmnopqrstuvwxz"  # pragma: allowlist secret
BIG_INSERT_ROWS = 5000


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _make_sqlite() -> SQLStorage:
    from trustyai_service.service.data.storage.sqlite.sqlite import (  # noqa: PLC0415
        SQLiteStorage,
    )

    return SQLiteStorage(":memory:")


def _make_postgres() -> SQLStorage:
    from trustyai_service.service.data.storage.postgres.postgres import (  # noqa: PLC0415
        PostgreSQLStorage,
    )

    return PostgreSQLStorage(
        "trustyai", "trustyai", "127.0.0.1", 5432, "trustyai-database"
    )


def _make_maria() -> SQLStorage:
    from trustyai_service.service.data.storage.maria.maria import (  # noqa: PLC0415
        MariaDBStorage,
    )

    return MariaDBStorage(
        "trustyai",
        "trustyai",
        "127.0.0.1",
        3306,
        "trustyai-database",
        attempt_migration=False,
    )


def _available_backends() -> list:
    # In-memory SQLite is independent per test, so it parallelizes freely. Live
    # backends share one server + the trustyai_v2 schema, so they must run
    # serially (same xdist group) to avoid cross-test clobbering.
    # Group names match the existing live suites (test_postgres_storage.py uses
    # "postgres", the MariaDB suites use "mariadb") so ALL tests touching a given
    # live server run serially in one xdist group and never clobber each other.
    backends = [pytest.param("sqlite")]
    if _port_open("127.0.0.1", 5432):
        backends.append(
            pytest.param("postgres", marks=pytest.mark.xdist_group("postgres"))
        )
    try:
        import mariadb  # noqa: F401, PLC0415

        if _port_open("127.0.0.1", 3306):
            backends.append(
                pytest.param("maria", marks=pytest.mark.xdist_group("mariadb"))
            )
    except ImportError:
        pass
    return backends


_FACTORIES = {
    "sqlite": _make_sqlite,
    "postgres": _make_postgres,
    "maria": _make_maria,
}


@pytest.fixture
def names() -> SQLTestResources:
    """Allocate resource names owned by this test."""
    return SQLTestResources()


@pytest.fixture(params=_available_backends())
def storage(
    request: pytest.FixtureRequest, names: SQLTestResources
) -> Iterator[SQLStorage]:
    """Yield a backend and clean up only this test's registered resources."""
    backend = _FACTORIES[request.param]()
    try:
        yield backend
    finally:
        try:
            asyncio.run(names.cleanup(backend))
        finally:
            backend._engine.dispose()


async def _store(
    storage: SQLStorage, names: SQLTestResources, seed: int, n_rows: int, n_cols: int
) -> tuple[np.ndarray, list[str], str]:
    dataset = np.arange(0, n_rows * n_cols).reshape(n_rows, n_cols)
    column_names = [ALPHABET[i] for i in range(n_cols)]
    dataset_name = names.dataset(f"dataset_{ALPHABET[seed]}")
    await storage.write_data(dataset_name, dataset, column_names)
    return dataset, column_names, dataset_name


@pytest.mark.asyncio
async def test_retrieve_full_and_partial(
    storage: SQLStorage, names: SQLTestResources
) -> None:
    """Full read plus a LIMIT/OFFSET window match the original array."""
    data, _, name = await _store(storage, names, 3, 9, 4)
    assert np.array_equal(await storage.read_data(name), data)
    assert await storage.dataset_shape(name) == data.shape
    assert await storage.dataset_rows(name) == data.shape[0]
    assert await storage.dataset_cols(name) == data.shape[1]
    assert np.array_equal(await storage.read_data(name, 2, 3), data[2:5])


@pytest.mark.asyncio
async def test_append(storage: SQLStorage, names: SQLTestResources) -> None:
    """Appending rows extends the dataset and preserves order."""
    data, cols, name = await _store(storage, names, 1, 5, 3)
    more = np.arange(100, 109).reshape(3, 3)
    await storage.write_data(name, more, cols)
    assert await storage.dataset_rows(name) == len(data) + len(more)
    assert np.array_equal(await storage.read_data(name), np.vstack([data, more]))


@pytest.mark.asyncio
async def test_append_with_reordered_columns_raises(
    storage: SQLStorage, names: SQLTestResources
) -> None:
    """Appending with the same names in a different order is refused."""
    _, cols, name = await _store(storage, names, 1, 4, 3)
    reordered = [cols[1], cols[0], cols[2]]
    with pytest.raises(ValueError, match="Column mismatch"):
        await storage.write_data(name, np.arange(3).reshape(1, 3), reordered)
    # The rejected append leaves the dataset untouched.
    assert await storage.dataset_rows(name) == 4


@pytest.mark.asyncio
async def test_append_with_renamed_column_raises(
    storage: SQLStorage, names: SQLTestResources
) -> None:
    """Appending under a different column name is refused."""
    _, cols, name = await _store(storage, names, 1, 2, 3)
    renamed = [*cols[:-1], "something_else"]
    with pytest.raises(ValueError, match="Column mismatch"):
        await storage.write_data(name, np.arange(3).reshape(1, 3), renamed)


@pytest.mark.asyncio
async def test_append_after_name_mapping_uses_original_names(
    storage: SQLStorage, names: SQLTestResources
) -> None:
    """An alias does not change which column names an append must supply."""
    data, cols, name = await _store(storage, names, 2, 2, 3)
    await storage.apply_name_mapping(name, {cols[0]: "aliased"})
    more = np.arange(100, 103).reshape(1, 3)
    await storage.write_data(name, more, cols)
    assert np.array_equal(await storage.read_data(name), np.vstack([data, more]))


@pytest.mark.asyncio
async def test_big_insert(storage: SQLStorage, names: SQLTestResources) -> None:
    """A 5000-row dataset round-trips."""
    data, _, name = await _store(storage, names, 0, BIG_INSERT_ROWS, 10)
    assert np.array_equal(await storage.read_data(name), data)
    assert await storage.dataset_rows(name) == BIG_INSERT_ROWS


@pytest.mark.asyncio
async def test_single_row(storage: SQLStorage, names: SQLTestResources) -> None:
    """A single-row dataset round-trips."""
    data, _, name = await _store(storage, names, 0, 1, 10)
    assert np.array_equal(await storage.read_data(name, 0, 1), data)


@pytest.mark.asyncio
async def test_vector_reshaped_to_column(
    storage: SQLStorage, names: SQLTestResources
) -> None:
    """A 1-D vector is stored as a single column."""
    data = np.arange(0, 10)
    name = names.dataset("vec")
    await storage.write_data(name, data, ["single_column"])
    assert np.array_equal((await storage.read_data(name)).reshape(-1), data)
    assert await storage.dataset_rows(name) == len(data)
    assert await storage.dataset_cols(name) == 1


@pytest.mark.asyncio
async def test_list_all_datasets(storage: SQLStorage, names: SQLTestResources) -> None:
    """All written dataset names are listed."""
    stored_names = set()
    for i in range(1, 5):
        _, _, name = await _store(storage, names, i, 3, 3)
        stored_names.add(name)
    listed = {
        name
        for name in await storage.list_all_datasets()
        if name.startswith(names.prefix)
    }
    assert listed == stored_names


@pytest.mark.asyncio
async def test_name_mapping_apply_and_clear(
    storage: SQLStorage, names: SQLTestResources
) -> None:
    """Aliases apply, originals persist, and clearing resets aliases."""
    _, cols, name = await _store(storage, names, 2, 4, 4)
    mapping = {c: "aliased_" + c for i, c in enumerate(cols) if i % 2 == 0}
    expected = [mapping.get(c, c) for c in cols]
    await storage.apply_name_mapping(name, mapping)
    assert await storage.get_original_column_names(name) == cols
    assert await storage.get_aliased_column_names(name) == expected
    await storage.clear_name_mapping(name)
    assert await storage.get_aliased_column_names(name) == cols


@pytest.mark.asyncio
async def test_delete_dataset(storage: SQLStorage, names: SQLTestResources) -> None:
    """A dataset can be deleted."""
    _, _, name = await _store(storage, names, 2, 3, 3)
    assert await storage.dataset_exists(name)
    await storage.delete_dataset(name)
    assert not await storage.dataset_exists(name)


@pytest.mark.asyncio
async def test_missing_dataset_raises(
    storage: SQLStorage, names: SQLTestResources
) -> None:
    """Operations on an unknown dataset raise ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        await storage.read_data(names.dataset("nope"))


@pytest.mark.asyncio
async def test_write_empty_raises(storage: SQLStorage, names: SQLTestResources) -> None:
    """Writing zero rows raises ValueError."""
    with pytest.raises(ValueError, match="No data provided"):
        await storage.write_data(names.dataset("empty"), np.array([]), ["a"])


@pytest.mark.asyncio
async def test_partial_payload_roundtrip(
    storage: SQLStorage, names: SQLTestResources
) -> None:
    """Partial-payload persist / get / delete round-trips."""
    payload_id = names.payload("req-1")
    payload = PartialPayload(data="dGVzdA==")
    await storage.persist_partial_payload(payload, payload_id, is_input=True)
    got = await storage.get_partial_payload(
        payload_id, is_input=True, is_modelmesh=True
    )
    assert got is not None
    assert got.data == payload.data
    await storage.delete_partial_payload(payload_id, is_input=True)
    assert (
        await storage.get_partial_payload(payload_id, is_input=True, is_modelmesh=True)
        is None
    )


@pytest.mark.asyncio
async def test_get_known_models_and_metadata(
    storage: SQLStorage, names: SQLTestResources
) -> None:
    """Known models are derived from dataset suffixes; metadata is assembled."""
    model_id = names.prefix + "mymodel"
    await storage.write_data(
        names.dataset("mymodel_inputs"), np.arange(6).reshape(2, 3), ["a", "b", "c"]
    )
    await storage.write_data(
        names.dataset("mymodel_outputs"), np.arange(2).reshape(2, 1), ["y"]
    )
    assert model_id in await storage.get_known_models()
    meta = await storage.get_metadata(model_id)
    assert meta["modelId"] == model_id
    assert meta["inputData"]["columnNames"] == ["a", "b", "c"]
    assert meta["outputData"]["shape"] == [2, 1]
