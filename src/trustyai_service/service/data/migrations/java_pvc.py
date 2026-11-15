"""Startup migration of PVC data written by the Java service.

The Java service stores per-model CSV files; the Python service reads HDF5 files
or a database. This module moves existing Java data across once, at startup,
before the API accepts traffic. The Java files are never modified or deleted.

This is a transition aid for the Java service. Remove it, ``pvc_csv.py``,
``java_values.py`` and ``pvc_target.py`` together once that service is out of
support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from trustyai_service.service.startup_migration_state import (
    StartupMigrationState,
    set_startup_migration_state,
)

from .cli import engine_options_from_environment, target_url_from_environment
from .errors import MigrationError, error_detail
from .pvc import PVCSource
from .pvc_csv import PVCCSVSource
from .pvc_target import PVCTarget, estimated_size
from .runner import MigrationConfig, MigrationResult, MigrationRunner, MigrationStatus
from .target import DatabaseTarget

if TYPE_CHECKING:
    from .contracts import MigrationSource, SourceManifest

logger = logging.getLogger(__name__)

_STATE_DIR = ".java-migration"
_STAGING_DIR = "staging"
_LEGACY_DIR = ".java-legacy"
_HDF5_STATE_FILE = "hdf5.json"
_DATABASE_STATE_FILE = "database.json"
_INTERNAL_SUFFIX = "-internal_data.csv"
_METADATA_SUFFIX = "-metadata.json"
_DEFAULT_DATA_FILE = "data.csv"
_DATABASE_NAMESPACE = "java-pvc"
_TRUTHY = ("1", "true", "yes", "on")
_PHASE_INSTALLING = "installing"
_PHASE_COMPLETE = "complete"


class JavaPVCMigrationError(MigrationError):
    """Raised when Java PVC data could not be migrated and the service must not start."""


@dataclass(frozen=True)
class _Plan:
    """Where the Java data is and where it goes."""

    folder: Path
    data_file: str
    database_url: str | None


async def run_startup_migrations() -> None:
    """Migrate Java PVC data when this start needs it, otherwise do nothing.

    Raises:
        JavaPVCMigrationError: The data could not be migrated. The caller must
            not start serving, because the service would run without its data.

    """
    plan = _plan_from_environment()
    if plan is None:
        return
    set_startup_migration_state(StartupMigrationState.RUNNING)
    try:
        await _run_off_loop(plan)
    except Exception:
        set_startup_migration_state(StartupMigrationState.FAILED)
        raise
    set_startup_migration_state(StartupMigrationState.DONE)


def has_java_pvc_data(folder: Path) -> bool:
    """Return whether the folder holds Java service per-model CSV files."""
    return folder.is_dir() and any(
        entry.is_file() and entry.name.endswith(_INTERNAL_SUFFIX)
        for entry in folder.iterdir()
    )


def _plan_from_environment() -> _Plan | None:
    """Choose a migration from the service configuration, or none."""
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "PVC").strip().upper()
    data_file = (
        os.environ.get("STORAGE_MIGRATION_CONFIG_FROM_FILENAME")
        or os.environ.get("STORAGE_DATA_FILENAME")
        or _DEFAULT_DATA_FILE
    )
    if storage_format == "PVC":
        folder = os.environ.get("STORAGE_DATA_FOLDER")
        return _Plan(Path(folder), data_file, None) if folder else None

    requested = os.environ.get("DATABASE_ATTEMPT_MIGRATION", "0").lower() in _TRUTHY
    folder = os.environ.get("STORAGE_MIGRATION_CONFIG_FROM_FOLDER") or os.environ.get(
        "STORAGE_DATA_FOLDER"
    )
    if not requested or not folder:
        return None
    database_url = target_url_from_environment()
    return _Plan(Path(folder), data_file, database_url) if database_url else None


async def _run_off_loop(plan: _Plan) -> None:
    """Run the migration in a child process so it cannot starve the health probes.

    The conversion is CPU-bound Python. In a thread it would hold the interpreter
    lock long enough for the liveness probe to time out and restart the pod.
    The child reads the same environment and logs to the same output.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", __name__, stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    if process.returncode != 0:
        lines = stdout.decode(errors="replace").strip().splitlines()
        msg = (
            lines[-1]
            if lines
            else f"Java PVC data in {plan.folder} could not be migrated "
            f"(exit code {process.returncode}, see the log)"
        )
        raise JavaPVCMigrationError(msg)


def main() -> int:
    """Child process entry point: run the planned migration, report one error line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    plan = _plan_from_environment()
    if plan is None:
        return 0
    try:
        asyncio.run(_execute(plan))
    except JavaPVCMigrationError as error:
        sys.stdout.write(f"{error}\n")
        logger.exception("Java PVC data migration failed")
        return 1
    return 0


async def _execute(plan: _Plan) -> None:
    """Run the migration the plan describes."""
    try:
        if plan.database_url is None:
            await _convert_to_hdf5(plan)
        else:
            await _copy_to_database(plan)
    except JavaPVCMigrationError:
        raise
    except MigrationError as error:
        msg = f"Java PVC data in {plan.folder} could not be migrated: {error}"
        raise JavaPVCMigrationError(msg) from error
    except SQLAlchemyError as error:
        # The driver's message can carry SQL parameters or credentials.
        msg = f"Java PVC data in {plan.folder} could not be migrated: {error_detail(error)}"
        raise JavaPVCMigrationError(msg) from None


async def _convert_to_hdf5(plan: _Plan) -> None:
    """Convert Java CSV files into the HDF5 files the service reads."""
    state_dir = plan.folder / _STATE_DIR
    state = _read_state(state_dir / _HDF5_STATE_FILE)
    if state is not None and state["phase"] == _PHASE_COMPLETE:
        return
    if state is None:
        if not has_java_pvc_data(plan.folder):
            return
        state = await _stage(plan, state_dir)
        _write_state(state_dir / _HDF5_STATE_FILE, state)
    _install(plan.folder, state)
    _write_state(state_dir / _HDF5_STATE_FILE, {**state, "phase": _PHASE_COMPLETE})
    logger.info("Java PVC data in %s is now in the service format", plan.folder)


async def _stage(plan: _Plan, state_dir: Path) -> dict[str, object]:
    """Convert and verify the data in a staging directory, leaving the source alone."""
    staging = state_dir / _STAGING_DIR
    shutil.rmtree(staging, ignore_errors=True)
    logger.info("Converting Java PVC data in %s", plan.folder)
    source = PVCCSVSource(plan.folder, quiesced=True, data_file=plan.data_file)
    _require_free_space(plan.folder, await source.preflight())
    try:
        result = await MigrationRunner(
            source=source,
            target=PVCTarget(staging, plan.data_file),
            config=MigrationConfig(),
        ).run()
        _require_complete(result, plan.folder)
    except Exception:
        # Free the space now; the next start would discard it anyway.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    java_files = sorted(
        entry.name
        for entry in plan.folder.iterdir()
        if entry.is_file()
        and (entry.suffix.lower() == ".csv" or entry.name.endswith(_METADATA_SUFFIX))
    )
    staged_files = sorted(entry.name for entry in staging.iterdir())
    overlap = set(java_files) & set(staged_files)
    if overlap:
        msg = f"converted files would overwrite Java files: {sorted(overlap)}"
        raise JavaPVCMigrationError(msg)
    return {
        "phase": _PHASE_INSTALLING,
        "java_files": java_files,
        "staged_files": staged_files,
    }


def _require_free_space(folder: Path, manifest: SourceManifest) -> None:
    """Fail before converting if the volume cannot hold the converted files.

    The Java files stay on the volume, and the converted files can be many
    times larger, so running out of space would otherwise be found minutes in.
    """
    needed = sum(estimated_size(dataset) for dataset in manifest.datasets)
    free = shutil.disk_usage(folder).free
    if needed > free:
        msg = (
            f"converting the Java data in {folder} needs about {needed} bytes but "
            f"only {free} are free; enlarge the volume or free space and restart"
        )
        raise JavaPVCMigrationError(msg)


def _install(folder: Path, state: dict[str, object]) -> None:
    """Move the Java files aside, then the converted files into place.

    Each step can be repeated, so a restart after a crash finishes the job.
    """
    staging = folder / _STATE_DIR / _STAGING_DIR
    legacy = folder / _LEGACY_DIR
    legacy.mkdir(exist_ok=True)
    for name in _names(state, "java_files"):
        if (folder / name).exists():
            (folder / name).replace(legacy / name)
    for name in _names(state, "staged_files"):
        if (staging / name).exists():
            (staging / name).replace(folder / name)
        elif not (folder / name).exists():
            msg = f"converted file {name} is missing from {staging}"
            raise JavaPVCMigrationError(msg)
    _sync_directory(folder)
    shutil.rmtree(staging, ignore_errors=True)


async def _copy_to_database(plan: _Plan) -> None:
    """Copy the Java data, in whichever form it now has, into the configured database."""
    if plan.database_url is None:
        msg = "no database target is configured"
        raise JavaPVCMigrationError(msg)
    marker = plan.folder / _STATE_DIR / _DATABASE_STATE_FILE
    target_id = _target_identity(plan.database_url)
    state = _read_state(marker)
    if state is not None and state.get("target") == target_id:
        return
    source = _database_source(plan)
    if source is None:
        return
    logger.info("Copying Java PVC data in %s to the database", plan.folder)
    engine = create_async_engine(
        plan.database_url, **engine_options_from_environment(plan.database_url)
    )
    try:
        result = await MigrationRunner(
            source=source,
            target=DatabaseTarget(engine, migration_namespace=_DATABASE_NAMESPACE),
            config=MigrationConfig(),
        ).run()
    finally:
        await engine.dispose()
    _require_complete(result, plan.folder)
    _write_state(marker, {"phase": _PHASE_COMPLETE, "target": target_id})
    logger.info("Java PVC data in %s is now in the database", plan.folder)


def _database_source(plan: _Plan) -> MigrationSource | None:
    """Pick the source holding the Java data: the CSV files, or their converted form.

    An instance that was first upgraded in place (converted to HDF5) and later
    moved to a database has no Java files left in the folder, but the converted
    files hold the same data plus anything written since.
    """
    if has_java_pvc_data(plan.folder):
        return PVCCSVSource(plan.folder, quiesced=True, data_file=plan.data_file)
    converted = _read_state(plan.folder / _STATE_DIR / _HDF5_STATE_FILE)
    if converted is None or converted["phase"] != _PHASE_COMPLETE:
        return None
    if any(plan.folder.glob("*.hdf5")):
        # Someone staged HDF5 files for the database migration that already
        # exists for them (and takes priority); do not import the data twice.
        logger.info(
            "HDF5 files are present in %s; leaving them to that migration", plan.folder
        )
        return None
    return PVCSource(plan.folder, quiesced=True, data_file=plan.data_file)


def _require_complete(result: MigrationResult, folder: Path) -> None:
    """Raise unless the runner finished and verified every dataset."""
    if result.status == MigrationStatus.COMPLETE:
        return
    detail = result.error or "; ".join(
        f"{failed.name}: {failed.error}" for failed in result.failed_datasets
    )
    msg = (
        f"Java PVC data in {folder} was not fully migrated ({result.status}): {detail}"
    )
    raise JavaPVCMigrationError(msg)


def _target_identity(database_url: str) -> str:
    """Name the target database without credentials."""
    url = make_url(database_url)
    return f"{url.drivername}://{url.host or ''}:{url.port or ''}/{url.database or ''}"


def _names(state: dict[str, object], key: str) -> list[str]:
    """Read a list of file names from a state record."""
    value = state[key]
    if not isinstance(value, list):
        msg = f"migration state field {key} is not a list"
        raise JavaPVCMigrationError(msg)
    return [str(name) for name in value]


def _read_state(path: Path) -> dict[str, object] | None:
    """Read a state record, or None when there is none."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        msg = f"cannot read migration state {path}: {error}"
        raise JavaPVCMigrationError(msg) from error
    if not isinstance(record, dict) or "phase" not in record:
        msg = f"migration state {path} is malformed"
        raise JavaPVCMigrationError(msg)
    return record


def _write_state(path: Path, record: dict[str, object]) -> None:
    """Write a state record so a reader sees the old or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(record))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _sync_directory(directory: Path) -> None:
    """Flush directory entries to disk so renames survive a crash."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
