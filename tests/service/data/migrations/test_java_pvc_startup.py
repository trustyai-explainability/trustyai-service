"""Tests for the startup migration of Java-written PVC data."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from trustyai_service.service.data.migrations import java_pvc
from trustyai_service.service.data.migrations.contracts import DatasetDescriptor
from trustyai_service.service.data.migrations.java_pvc import (
    JavaPVCMigrationError,
    run_startup_migrations,
)
from trustyai_service.service.data.migrations.pvc_target import estimated_size
from trustyai_service.service.data.migrations.runner import (
    MigrationResult,
    MigrationStatus,
)
from trustyai_service.service.data.storage.pvc import PVCStorage
from trustyai_service.service.health_checks import check_startup_migration_readiness
from trustyai_service.service.startup_migration_state import (
    StartupMigrationState,
    get_startup_migration_state,
    set_startup_migration_state,
)

from .test_pvc_csv_source import _write_java_csv_model


class _Usage(NamedTuple):
    """Stand-in for the result of ``shutil.disk_usage``."""

    total: int
    used: int
    free: int


_JAVA_FILES = {
    "model-42-data.csv",
    "model-42-internal_data.csv",
    "model-42-metadata.json",
}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every test no migration state and no storage configuration, and clear it after."""
    for name in (
        "SERVICE_STORAGE_FORMAT",
        "STORAGE_DATA_FOLDER",
        "STORAGE_DATA_FILENAME",
        "STORAGE_MIGRATION_CONFIG_FROM_FOLDER",
        "STORAGE_MIGRATION_CONFIG_FROM_FILENAME",
        "DATABASE_ATTEMPT_MIGRATION",
        "STORAGE_DATABASE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    set_startup_migration_state(StartupMigrationState.IDLE)
    yield
    set_startup_migration_state(StartupMigrationState.IDLE)


@pytest.fixture
def real_subprocess() -> None:
    """Ask a test to run the migration in a real child process, as the service does."""


@pytest.fixture(autouse=True)
def _in_process_migration(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the migration in this process, so tests can patch its internals and stay fast."""
    if "real_subprocess" in request.fixturenames:
        return

    async def in_process(plan: java_pvc._Plan) -> None:
        await java_pvc._execute(plan)

    monkeypatch.setattr(java_pvc, "_run_off_loop", in_process)


def _mtimes(folder: Path) -> dict[str, int]:
    """Map each file in the folder to its modification time."""
    return {
        path.name: path.stat().st_mtime_ns
        for path in folder.iterdir()
        if path.is_file()
    }


def _contents(folder: Path) -> dict[str, bytes]:
    """Map each file in the folder to its bytes."""
    return {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}


def _names(folder: Path, pattern: str = "*") -> set[str]:
    """List the names in a folder that match a pattern."""
    return {path.name for path in folder.glob(pattern)}


def _pvc_env(monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
    """Configure the service the way the operator does for a PVC instance."""
    monkeypatch.setenv("SERVICE_STORAGE_FORMAT", "PVC")
    monkeypatch.setenv("STORAGE_DATA_FOLDER", str(folder))
    monkeypatch.setenv("STORAGE_DATA_FILENAME", "data.csv")


def _write_numeric_model(folder: Path) -> None:
    """Write a Java model of integer inputs and float outputs only."""
    schema = {
        "items": {
            "a": {"type": "INT32", "name": "a", "columnIndex": 0},
            "b": {"type": "INT64", "name": "b", "columnIndex": 1},
        }
    }
    output = {"items": {"p": {"type": "DOUBLE", "name": "p", "columnIndex": 2}}}
    (folder / "nums-data.csv").write_text(
        "1,9223372036854775807,0.5\n-2,-9223372036854775808,1.5\n", encoding="utf-8"
    )
    (folder / "nums-internal_data.csv").write_text(
        "organic,id-1,2024-02-03T04:05:06\nsynthetic,id-2,2024-02-03T04:06:07\n",
        encoding="utf-8",
    )
    (folder / "nums-metadata.json").write_text(
        json.dumps(
            {
                "modelId": "nums",
                "observations": 2,
                "recordedInferences": True,
                "inputSchema": schema,
                "outputSchema": output,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("real_subprocess")
async def test_converts_java_data_into_the_service_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service reads the Java data through its normal storage after start."""
    _write_java_csv_model(tmp_path)
    _pvc_env(monkeypatch, tmp_path)

    await run_startup_migrations()

    assert get_startup_migration_state() == StartupMigrationState.DONE
    storage = PVCStorage(str(tmp_path), "data.csv")
    assert await storage.get_known_models() == ["model-42"]
    assert (await storage.read_data("model-42_outputs")).tolist() == [[0.75], [0.25]]
    inputs = await storage.read_data("model-42_inputs")
    assert inputs[0][0] == "hello, world"
    assert inputs[1][4] == -2147483648
    assert await storage.get_original_column_names("model-42_outputs") == ["score"]
    assert await storage.get_aliased_column_names("model-42_outputs") == ["risk"]
    assert (await storage.get_aliased_column_names("model-42_inputs"))[0] == "prompt"
    assert (await storage.get_column_types("model-42_inputs"))[1] == "TENSOR"
    metadata = await storage.read_data("model-42_metadata")
    assert metadata.tolist() == [
        ["2024-02-03T04:05:06.123456789", "id-1", "organic"],
        ["2024-02-03T04:06:07", "id-2", "synthetic"],
    ]
    assert await storage.dataset_rows("model-42_inputs") == 2


@pytest.mark.asyncio
async def test_keeps_the_java_files_and_leaves_the_folder_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Java files move to a legacy folder unchanged; nothing else is left behind."""
    _write_java_csv_model(tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in _JAVA_FILES}
    _pvc_env(monkeypatch, tmp_path)

    await run_startup_migrations()

    legacy = tmp_path / ".java-legacy"
    assert {path.name for path in legacy.iterdir()} == _JAVA_FILES
    assert {name: (legacy / name).read_bytes() for name in _JAVA_FILES} == before
    assert not any((tmp_path / name).exists() for name in _JAVA_FILES)
    assert not (tmp_path / ".java-migration" / "staging").exists()
    assert not (tmp_path / "model-42_outputs_data.csv.tmp").exists()


@pytest.mark.asyncio
async def test_second_start_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart after a completed conversion does not touch the data."""
    _write_java_csv_model(tmp_path)
    _pvc_env(monkeypatch, tmp_path)
    await run_startup_migrations()
    snapshot = _mtimes(tmp_path)

    await run_startup_migrations()

    assert _mtimes(tmp_path) == snapshot


@pytest.mark.asyncio
async def test_new_data_written_after_conversion_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows the service adds later are not replaced by a repeat start."""
    _write_java_csv_model(tmp_path)
    _pvc_env(monkeypatch, tmp_path)
    await run_startup_migrations()
    storage = PVCStorage(str(tmp_path), "data.csv")
    await storage.write_data("model-42_outputs", np.array([[0.5]]), ["score"])

    await run_startup_migrations()

    assert (
        await PVCStorage(str(tmp_path), "data.csv").dataset_rows("model-42_outputs")
        == 3
    )


@pytest.mark.asyncio
async def test_stores_integer_and_float_datasets_as_plain_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integer columns keep full 64-bit precision."""
    _write_numeric_model(tmp_path)
    _pvc_env(monkeypatch, tmp_path)

    await run_startup_migrations()

    storage = PVCStorage(str(tmp_path), "data.csv")
    rows = await storage.read_data("nums_inputs")
    assert rows.dtype == np.int64
    assert rows.tolist() == [[1, 9223372036854775807], [-2, -9223372036854775808]]
    assert (await storage.read_data("nums_outputs")).tolist() == [[0.5], [1.5]]


@pytest.mark.asyncio
async def test_does_nothing_without_java_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder with no Java files is left exactly as it is."""
    _pvc_env(monkeypatch, tmp_path)

    await run_startup_migrations()

    assert _names(tmp_path) == set()
    assert get_startup_migration_state() == StartupMigrationState.DONE


@pytest.mark.asyncio
async def test_does_nothing_when_storage_folder_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a folder there is nothing to inspect, and probes stay ready."""
    monkeypatch.setenv("SERVICE_STORAGE_FORMAT", "PVC")

    await run_startup_migrations()

    assert get_startup_migration_state() == StartupMigrationState.IDLE


@pytest.mark.asyncio
@pytest.mark.usefixtures("real_subprocess")
async def test_failure_blocks_startup_and_leaves_the_source_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupt Java data stops the service and is not modified."""
    metadata_path = _write_java_csv_model(tmp_path)
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["observations"] = 99
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    before = _contents(tmp_path)
    _pvc_env(monkeypatch, tmp_path)

    with pytest.raises(JavaPVCMigrationError, match="could not be migrated"):
        await run_startup_migrations()

    assert get_startup_migration_state() == StartupMigrationState.FAILED
    assert check_startup_migration_readiness().status == "error"
    assert not (tmp_path / ".java-legacy").exists()
    assert _contents(tmp_path) == before


@pytest.mark.asyncio
async def test_existing_service_files_alongside_java_data_fail_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder the service already wrote to is not merged silently."""
    _write_java_csv_model(tmp_path)
    storage = PVCStorage(str(tmp_path), "data.csv")
    await storage.write_data("model-42_outputs", np.array([[1.0]]), ["score"])
    _pvc_env(monkeypatch, tmp_path)

    with pytest.raises(JavaPVCMigrationError, match="unselected storage file"):
        await run_startup_migrations()

    assert _names(tmp_path, "model-42-*") == _JAVA_FILES


@pytest.mark.asyncio
async def test_leftover_staging_from_a_crashed_attempt_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash before the install phase starts over from the intact source."""
    _write_java_csv_model(tmp_path)
    staging = tmp_path / ".java-migration" / "staging"
    staging.mkdir(parents=True)
    (staging / "garbage_inputs_data.csv").write_bytes(b"partial")
    _pvc_env(monkeypatch, tmp_path)

    await run_startup_migrations()

    assert not (tmp_path / "garbage_inputs_data.csv").exists()
    storage = PVCStorage(str(tmp_path), "data.csv")
    assert await storage.get_known_models() == ["model-42"]


@pytest.mark.asyncio
@pytest.mark.parametrize("moved_java_files", [0, 1, 3], ids=["none", "one", "all"])
async def test_crash_during_install_is_finished_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, moved_java_files: int
) -> None:
    """A restart completes an interrupted install without converting again."""
    _write_java_csv_model(tmp_path)
    plan = java_pvc._Plan(tmp_path, "data.csv", None)
    state_dir = tmp_path / ".java-migration"
    state = await java_pvc._stage(plan, state_dir)
    java_pvc._write_state(state_dir / "hdf5.json", state)
    legacy = tmp_path / ".java-legacy"
    legacy.mkdir()
    for name in sorted(_JAVA_FILES)[:moved_java_files]:
        (tmp_path / name).replace(legacy / name)
    _pvc_env(monkeypatch, tmp_path)

    async def _fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the data must not be converted again")

    monkeypatch.setattr(java_pvc, "_stage", _fail)

    await run_startup_migrations()

    storage = PVCStorage(str(tmp_path), "data.csv")
    assert await storage.get_known_models() == ["model-42"]
    assert {path.name for path in legacy.iterdir()} == _JAVA_FILES


@pytest.mark.asyncio
async def test_install_reports_a_converted_file_that_vanished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An install that cannot find a converted file stops instead of continuing."""
    _write_java_csv_model(tmp_path)
    plan = java_pvc._Plan(tmp_path, "data.csv", None)
    state_dir = tmp_path / ".java-migration"
    state = await java_pvc._stage(plan, state_dir)
    java_pvc._write_state(state_dir / "hdf5.json", state)
    next((state_dir / "staging").iterdir()).unlink()
    _pvc_env(monkeypatch, tmp_path)

    with pytest.raises(JavaPVCMigrationError, match="is missing"):
        await run_startup_migrations()


@pytest.mark.asyncio
async def test_malformed_state_file_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged state record is reported, not guessed at."""
    _write_java_csv_model(tmp_path)
    (tmp_path / ".java-migration").mkdir()
    (tmp_path / ".java-migration" / "hdf5.json").write_text("{", encoding="utf-8")
    _pvc_env(monkeypatch, tmp_path)

    with pytest.raises(JavaPVCMigrationError, match="cannot read migration state"):
        await run_startup_migrations()


def _database_env(
    monkeypatch: pytest.MonkeyPatch, folder: Path, database: Path
) -> None:
    """Configure the operator's PVC to database migration against SQLite."""
    monkeypatch.setenv("SERVICE_STORAGE_FORMAT", "SQLITE")
    monkeypatch.setenv("STORAGE_DATABASE_PATH", str(database))
    monkeypatch.setenv("DATABASE_ATTEMPT_MIGRATION", "true")
    monkeypatch.setenv("STORAGE_MIGRATION_CONFIG_FROM_FOLDER", str(folder))
    monkeypatch.setenv("STORAGE_MIGRATION_CONFIG_FROM_FILENAME", "data.csv")


@pytest.mark.asyncio
@pytest.mark.usefixtures("real_subprocess")
async def test_copies_java_data_into_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's PVC to database flow moves Java data into the database."""
    from trustyai_service.service.data.storage.sqlite.sqlite import SQLiteStorage

    folder = tmp_path / "pvc"
    _write_java_csv_model(folder)
    database = tmp_path / "service.sqlite"
    _database_env(monkeypatch, folder, database)

    await run_startup_migrations()

    storage = SQLiteStorage(str(database))
    try:
        assert (await storage.read_data("model-42_outputs")).tolist() == [
            [0.75],
            [0.25],
        ]
        assert await storage.dataset_rows("model-42_inputs") == 2
    finally:
        storage._engine.dispose()
    assert {path.name for path in folder.glob("model-42-*")} == _JAVA_FILES
    marker = json.loads((folder / ".java-migration" / "database.json").read_text())
    assert marker["phase"] == "complete"
    assert "password" not in json.dumps(marker)


@pytest.mark.asyncio
async def test_database_copy_is_skipped_once_done_even_after_new_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows the service adds later do not make a repeat start fail."""
    from trustyai_service.service.data.storage.sqlite.sqlite import SQLiteStorage

    folder = tmp_path / "pvc"
    _write_java_csv_model(folder)
    database = tmp_path / "service.sqlite"
    _database_env(monkeypatch, folder, database)
    await run_startup_migrations()
    storage = SQLiteStorage(str(database))
    try:
        await storage.write_data("model-42_outputs", np.array([[0.5]]), ["score"])
    finally:
        storage._engine.dispose()

    await run_startup_migrations()

    storage = SQLiteStorage(str(database))
    try:
        assert await storage.dataset_rows("model-42_outputs") == 3
    finally:
        storage._engine.dispose()


@pytest.mark.asyncio
async def test_database_copy_needs_the_migration_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database instance that did not ask for migration is left alone."""
    folder = tmp_path / "pvc"
    _write_java_csv_model(folder)
    _database_env(monkeypatch, folder, tmp_path / "service.sqlite")
    monkeypatch.delenv("DATABASE_ATTEMPT_MIGRATION")

    await run_startup_migrations()

    assert get_startup_migration_state() == StartupMigrationState.IDLE
    assert not (folder / ".java-migration").exists()


@pytest.mark.asyncio
async def test_database_copy_failure_does_not_write_the_done_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed copy is retried on the next start."""
    folder = tmp_path / "pvc"
    metadata_path = _write_java_csv_model(folder)
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["observations"] = 99
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    _database_env(monkeypatch, folder, tmp_path / "service.sqlite")

    with pytest.raises(JavaPVCMigrationError):
        await run_startup_migrations()

    assert not (folder / ".java-migration" / "database.json").exists()


def test_readiness_follows_the_migration_state() -> None:
    """The probe is not ready while migrating or after a failure."""
    for state, expected in (
        (StartupMigrationState.IDLE, "ok"),
        (StartupMigrationState.RUNNING, "error"),
        (StartupMigrationState.FAILED, "error"),
        (StartupMigrationState.DONE, "ok"),
    ):
        set_startup_migration_state(state)
        assert check_startup_migration_readiness().status == expected


@pytest.mark.asyncio
async def test_converts_a_model_without_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model with no rows still becomes a readable, empty dataset."""
    (tmp_path / "empty-data.csv").write_text("", encoding="utf-8")
    (tmp_path / "empty-internal_data.csv").write_text("", encoding="utf-8")
    (tmp_path / "empty-metadata.json").write_text(
        json.dumps(
            {
                "modelId": "empty",
                "observations": 0,
                "recordedInferences": False,
                "inputSchema": {
                    "items": {"a": {"type": "STRING", "name": "a", "columnIndex": 0}}
                },
                "outputSchema": {
                    "items": {"p": {"type": "DOUBLE", "name": "p", "columnIndex": 1}}
                },
            }
        ),
        encoding="utf-8",
    )
    _pvc_env(monkeypatch, tmp_path)

    await run_startup_migrations()

    storage = PVCStorage(str(tmp_path), "data.csv")
    assert await storage.get_known_models() == ["empty"]
    assert await storage.dataset_rows("empty_inputs") == 0
    assert await storage.dataset_rows("empty_outputs") == 0


@pytest.mark.asyncio
async def test_failed_conversion_leaves_no_staging_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after data was staged frees that space immediately."""
    _write_java_csv_model(tmp_path)
    _pvc_env(monkeypatch, tmp_path)

    async def partial_run(_self: object) -> MigrationResult:
        staging = tmp_path / ".java-migration" / "staging"
        staging.mkdir(parents=True)
        (staging / "model-42_inputs_data.csv").write_bytes(b"partial")
        return MigrationResult(status=MigrationStatus.PARTIAL, error="disk full")

    monkeypatch.setattr(java_pvc.MigrationRunner, "run", partial_run)

    with pytest.raises(JavaPVCMigrationError, match="disk full"):
        await run_startup_migrations()

    assert not (tmp_path / ".java-migration" / "staging").exists()
    assert not (tmp_path / ".java-migration" / "hdf5.json").exists()


@pytest.mark.asyncio
async def test_database_copy_after_an_in_place_conversion_includes_new_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upgrade in place first, move to a database later: nothing is left behind."""
    from trustyai_service.service.data.storage.sqlite.sqlite import SQLiteStorage

    folder = tmp_path / "pvc"
    _write_java_csv_model(folder)
    _pvc_env(monkeypatch, folder)
    await run_startup_migrations()
    pvc = PVCStorage(str(folder), "data.csv")
    await pvc.write_data("model-42_outputs", np.array([[0.5]]), ["score"])
    await pvc.write_data(
        "model-42_inputs",
        [["later", [1.0], True, 0.5, 1, 2, {"rank": 3}, "x"]],
        list(await pvc.get_original_column_names("model-42_inputs")),
    )
    database = tmp_path / "service.sqlite"
    _database_env(monkeypatch, folder, database)
    monkeypatch.setenv("SERVICE_STORAGE_FORMAT", "SQLITE")

    await run_startup_migrations()

    storage = SQLiteStorage(str(database))
    try:
        assert await storage.dataset_rows("model-42_outputs") == 3
        assert await storage.dataset_rows("model-42_inputs") == 3
    finally:
        storage._engine.dispose()


@pytest.mark.asyncio
async def test_database_copy_leaves_staged_hdf5_files_to_the_existing_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HDF5 files staged for the existing migration are not imported a second time."""
    folder = tmp_path / "pvc"
    _write_java_csv_model(folder)
    _pvc_env(monkeypatch, folder)
    await run_startup_migrations()
    (folder / "model-42_trustyai.hdf5").write_bytes(b"staged")
    database = tmp_path / "service.sqlite"
    _database_env(monkeypatch, folder, database)
    monkeypatch.setenv("SERVICE_STORAGE_FORMAT", "SQLITE")

    await run_startup_migrations()

    assert not database.exists()
    assert not (folder / ".java-migration" / "database.json").exists()


@pytest.mark.asyncio
async def test_not_enough_free_space_fails_before_any_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A volume that cannot hold the converted files is reported up front."""
    _write_java_csv_model(tmp_path)
    _pvc_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        java_pvc.shutil, "disk_usage", lambda _path: _Usage(total=10, used=9, free=1)
    )

    with pytest.raises(JavaPVCMigrationError, match="only 1 are free"):
        await run_startup_migrations()

    assert not (tmp_path / ".java-migration").exists()
    assert not (tmp_path / ".java-legacy").exists()


def test_size_estimate_matches_how_each_dataset_is_stored() -> None:
    """Plain numeric datasets are estimated by cell, serialized ones by row."""
    plain = DatasetDescriptor(
        name="m_outputs",
        column_names=("a", "b"),
        aliases=("a", "b"),
        column_types=("FLOAT", "DOUBLE"),
        row_count=10,
    )
    mixed = DatasetDescriptor(
        name="m_inputs",
        column_names=("a", "b"),
        aliases=("a", "b"),
        column_types=("INT32", "DOUBLE"),
        row_count=10,
    )
    assert estimated_size(plain) == 10 * 2 * 8
    assert estimated_size(mixed) == 10 * 1024


@pytest.mark.asyncio
@pytest.mark.usefixtures("real_subprocess")
async def test_child_process_that_dies_without_a_message_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that exits without explaining still stops the service with a message."""
    _write_java_csv_model(tmp_path)
    _pvc_env(monkeypatch, tmp_path)
    monkeypatch.setattr(java_pvc.sys, "executable", shutil.which("false"))

    with pytest.raises(JavaPVCMigrationError, match=r"exit code 1, see the log"):
        await run_startup_migrations()

    assert get_startup_migration_state() == StartupMigrationState.FAILED
    assert _names(tmp_path, "model-42-*") == _JAVA_FILES
