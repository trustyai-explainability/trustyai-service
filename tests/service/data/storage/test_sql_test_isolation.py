"""Run the live suites with existing data in a disposable SQL database."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("sqlalchemy")

from tests.service.data.storage import test_sql_parity as parity
from tests.service.data.storage.sql_test_resources import SQLTestResources
from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.data.storage.sqlite.sqlite import SQLiteStorage

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterator
    from pathlib import Path
    from typing import Any


_LEGACY_NAMES = [
    *(f"dataset_{letter}" for letter in "abcdefghij"),
    "dataset_single_row",
    "vec",
    "mymodel_inputs",
    "mymodel_outputs",
    "nope",
    "empty",
]
_PAYLOAD_IDS = ["req-1", "req-123"]
_SENTINEL = np.array([[101, 102, 103]])
_COLUMNS = ["a", "b", "c"]
_ALIASES = ["existing_alias", "b", "c"]
_PAYLOAD = PartialPayload(data="cHJlZXhpc3Rpbmc=")
_PARITY_SCENARIOS = [
    scenario
    for name, scenario in vars(parity).items()
    if name.startswith("test_") and callable(scenario)
]


async def _seed_dataset(storage: SQLiteStorage, name: str) -> None:
    await storage.write_data(name, _SENTINEL, _COLUMNS)
    await storage.apply_name_mapping(name, {"a": _ALIASES[0]})


async def _seed_existing(storage: SQLiteStorage) -> None:
    for name in _LEGACY_NAMES:
        await _seed_dataset(storage, name)
    for payload_id in _PAYLOAD_IDS:
        for is_input in (True, False):
            await storage.persist_partial_payload(
                _PAYLOAD, payload_id, is_input=is_input
            )


async def _assert_existing_unchanged(storage: SQLiteStorage) -> None:
    expected_names = {*_LEGACY_NAMES, "created_during_test"}
    assert set(await storage.list_all_datasets()) == expected_names
    for name in expected_names:
        assert np.array_equal(await storage.read_data(name), _SENTINEL)
        assert await storage.dataset_shape(name) == _SENTINEL.shape
        assert await storage.get_original_column_names(name) == _COLUMNS
        assert await storage.get_aliased_column_names(name) == _ALIASES
    for payload_id in _PAYLOAD_IDS:
        for is_input in (True, False):
            payload = await storage.get_partial_payload(
                payload_id, is_input=is_input, is_modelmesh=True
            )
            assert payload == _PAYLOAD
    with storage._engine.connect() as conn:
        # No test-owned payload may survive cleanup, including failure paths.
        assert len(conn.execute(storage._payloads.select()).fetchall()) == 4


@pytest.fixture
def existing_storage(tmp_path: Path) -> Iterator[SQLiteStorage]:
    """Seed old test names without ever connecting to a developer's database."""
    storage = SQLiteStorage(str(tmp_path / "existing.sqlite"))
    try:
        asyncio.run(_seed_existing(storage))
        yield storage
    finally:
        storage._engine.dispose()


@pytest.mark.parametrize("scenario", _PARITY_SCENARIOS, ids=lambda test: test.__name__)
def test_parity_preserves_existing_data(
    existing_storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
    scenario: Callable[..., Coroutine[Any, Any, None]],
) -> None:
    """Every parity scenario and its fixture cleanup preserve other owners."""
    monkeypatch.setitem(parity._FACTORIES, "postgres", lambda: existing_storage)
    names = SQLTestResources()
    fixture = parity.storage.__wrapped__(SimpleNamespace(param="postgres"), names)
    storage = next(fixture)
    try:
        asyncio.run(_seed_dataset(storage, "created_during_test"))
        asyncio.run(scenario(storage, names))
    finally:
        fixture.close()
    asyncio.run(_assert_existing_unchanged(existing_storage))


@pytest.mark.parametrize(
    "scenario",
    [
        "test_retrieve_data",
        "test_name_mapping",
        "test_clear_name_mapping",
        "test_list_all_datasets",
        "test_delete_dataset",
        "test_partial_payload",
        "test_big_insert",
        "test_single_row_insert",
        "test_single_row_retrieval",
    ],
)
def test_postgres_suite_preserves_existing_data(
    existing_storage: SQLiteStorage, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """Run PostgreSQL test bodies and cleanup against an isolated database."""
    postgres = pytest.importorskip("tests.service.data.test_postgres_storage")
    monkeypatch.setattr(postgres, "PostgreSQLStorage", lambda *_: existing_storage)
    case = postgres.TestPostgreSQLStorage(methodName=scenario)
    case.setUp()
    try:
        asyncio.run(_seed_dataset(existing_storage, "created_during_test"))
        getattr(case, scenario)()
    finally:
        assert case.doCleanups()
    asyncio.run(_assert_existing_unchanged(existing_storage))


def test_cleanup_after_failed_payload_test(
    existing_storage: SQLiteStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture finalization removes incomplete test resources after failure."""
    monkeypatch.setitem(parity._FACTORIES, "postgres", lambda: existing_storage)
    names = SQLTestResources()
    fixture = parity.storage.__wrapped__(SimpleNamespace(param="postgres"), names)
    storage = next(fixture)
    asyncio.run(_seed_dataset(storage, "created_during_test"))
    asyncio.run(_seed_dataset(storage, names.dataset("unfinished")))
    asyncio.run(
        storage.persist_partial_payload(
            _PAYLOAD, names.payload("unfinished"), is_input=True
        )
    )
    with pytest.raises(AssertionError, match="simulated assertion failure"):
        fixture.throw(AssertionError("simulated assertion failure"))
    asyncio.run(_assert_existing_unchanged(existing_storage))
