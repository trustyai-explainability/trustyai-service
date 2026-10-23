"""Tests for migration lifecycle, safety, and resumability.

These tests intentionally use the public migration API rather than the
database schema. They should remain valid if MariaDB and PostgreSQL storage
internals change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from .conftest import result_status, target_storage


def _runner(api, source, target, **config_overrides):
    """Construct the runner using the deliberately small public API."""
    defaults = {
        "batch_size": 2,
        "require_quiescent_source": True,
        "resume": True,
        "verify_checksums": True,
    }
    defaults.update(config_overrides)
    config = api.MigrationConfig(**defaults)
    return api.MigrationRunner(source=source, target=target, config=config)


def test_public_api_is_available(migration_api) -> None:
    """The framework exposes one stable entry point for all backends."""
    assert migration_api.MigrationRunner is not None
    assert migration_api.MigrationConfig is not None
    assert migration_api.PVCSource is not None
    assert migration_api.PVCCSVSource is not None
    assert migration_api.MariaDBSource is not None
    assert migration_api.DatabaseTarget is not None


@pytest.mark.asyncio
async def test_preflight_rejects_non_quiesced_source(
    migration_api, pvc_fixture, target_factory
) -> None:
    """A migration cannot promise consistency while writes continue."""
    source = migration_api.PVCSource(
        pvc_fixture.directory,
        quiesced=False,
    )
    target = await target_factory("postgres")
    runner = _runner(migration_api, source, target)

    with pytest.raises(migration_api.MigrationPreflightError, match="quiesc"):
        await runner.preflight()


@pytest.mark.asyncio
async def test_preflight_rejects_missing_pvc_directory(
    migration_api, tmp_path, target_factory
) -> None:
    """A missing source cannot be treated as an empty successful migration."""
    source = migration_api.PVCSource(tmp_path / "does-not-exist", quiesced=True)
    target = await target_factory("postgres")
    runner = _runner(migration_api, source, target)

    with pytest.raises(migration_api.MigrationPreflightError, match="source"):
        await runner.preflight()


@pytest.mark.asyncio
async def test_invalid_batch_size_is_rejected(
    migration_api, pvc_fixture, target_factory
) -> None:
    """A non-positive batch size cannot produce a safe stream."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory("postgres")

    with pytest.raises(ValueError, match="batch"):
        _runner(migration_api, source, target, batch_size=0)


@pytest.mark.asyncio
async def test_dry_run_ignores_non_hdf5_files(
    migration_api, pvc_fixture, target_factory
) -> None:
    """Only recognized PVC files enter the logical migration manifest."""
    (pvc_fixture.directory / "notes.txt").write_text("not a dataset", encoding="utf-8")
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory("postgres")

    result = await _runner(migration_api, source, target, dry_run=True).run()

    assert result_status(result) == "DRY_RUN"
    assert result.total_files == 3
    assert result.total_datasets == 3


@pytest.mark.asyncio
async def test_dry_run_discovers_and_validates_without_writing(
    migration_api, pvc_fixture, target_factory
) -> None:
    """Dry-run produces a manifest but does not create target datasets."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory("postgres")
    runner = _runner(migration_api, source, target, dry_run=True)

    result = await runner.run()

    assert result_status(result) == "DRY_RUN"
    assert result.total_files == 3
    assert result.total_datasets == 3
    assert result.total_rows == 6
    assert await target_storage(target).list_all_datasets() == []


@pytest.mark.asyncio
async def test_failed_dataset_is_rolled_back_and_marked_for_resume(
    migration_api, pvc_fixture, failing_target_factory
) -> None:
    """A failed dataset cannot be reported complete or leave duplicate rows."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await failing_target_factory("postgres", fail_after_batches=1)
    runner = _runner(migration_api, source, target)

    result = await runner.run()

    assert result_status(result) in {"PARTIAL", "FAILED"}
    assert result.failed_datasets
    assert all(item.status != "COMPLETE" for item in result.failed_datasets)
    assert await target_storage(target).dataset_rows(pvc_fixture.datasets[0].name) == 0


@pytest.mark.asyncio
async def test_resume_retries_only_incomplete_work(
    migration_api, pvc_fixture, failing_target_factory
) -> None:
    """A second run resumes failed work without appending duplicate rows."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await failing_target_factory("postgres", fail_after_batches=1)
    first = await _runner(migration_api, source, target).run()
    assert result_status(first) in {"PARTIAL", "FAILED"}

    target.clear_failure()
    second = await _runner(migration_api, source, target).run()

    assert result_status(second) == "COMPLETE"
    storage = target_storage(target)
    for dataset in pvc_fixture.datasets:
        assert await storage.dataset_rows(dataset.name) == len(dataset.rows)


@pytest.mark.asyncio
async def test_repeated_run_is_idempotent(migration_api, pvc_fixture, target_factory):
    """Running the same source twice does not duplicate logical rows."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory("postgres")
    first = await _runner(migration_api, source, target).run()
    second = await _runner(migration_api, source, target).run()

    assert result_status(first) == "COMPLETE"
    assert result_status(second) == "COMPLETE"
    assert second.rows_written == 0
    for dataset in pvc_fixture.datasets:
        assert await target_storage(target).dataset_rows(dataset.name) == len(
            dataset.rows
        )


@pytest.mark.asyncio
async def test_source_files_are_never_deleted(
    migration_api, pvc_fixture, target_factory
) -> None:
    """Successful migration leaves the PVC available for audit and rollback."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    target = await target_factory("mariadb")
    result = await _runner(migration_api, source, target).run()

    assert result_status(result) == "COMPLETE"
    assert _hash_files(pvc_fixture.directory) == pvc_fixture.file_hashes


def _hash_files(directory: Path) -> dict[str, str]:
    """Hash helper kept local to avoid making source internals public."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.hdf5"))
    }
