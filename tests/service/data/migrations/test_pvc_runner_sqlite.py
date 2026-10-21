"""Local runner coverage using SQLite as a dialect-neutral test target."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from trustyai_service.service.data.storage.pvc import PARTIAL_INPUT_NAME
from trustyai_service.service.serialization import serialize_model

from .conftest import assert_dataset_matches, result_status, target_storage
from .test_migration_contract import _runner


@pytest.mark.asyncio
async def test_pvc_runner_transfers_all_datasets_to_sqlite(
    migration_api, pvc_fixture, sqlite_target
) -> None:
    """The canonical runner transfers rows and metadata through one target API."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)

    result = await _runner(migration_api, source, sqlite_target).run()

    assert result_status(result) == "COMPLETE"
    assert result.datasets_written == 3
    assert result.rows_written == 6
    for expected in pvc_fixture.datasets:
        await assert_dataset_matches(sqlite_target, expected)


@pytest.mark.asyncio
async def test_pvc_runner_rolls_back_and_resumes_on_target_failure(
    migration_api, pvc_fixture, sqlite_target
) -> None:
    """A failed batch leaves no partial dataset and a retry completes it."""
    failing_target = migration_api.DatabaseTarget(
        sqlite_target.engine,
        migration_namespace="sqlite_failure",
        fail_after_batches=1,
    )
    await failing_target.reset_for_test()
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)

    first = await _runner(migration_api, source, failing_target).run()

    assert result_status(first) == "PARTIAL"
    assert first.failed_datasets
    assert first.rows_written == 0
    storage = target_storage(failing_target)
    assert await storage.dataset_rows(pvc_fixture.datasets[0].name) == 0
    assert await storage.dataset_rows(pvc_fixture.datasets[1].name) == 0
    assert await storage.dataset_rows(pvc_fixture.datasets[2].name) == 0

    failing_target.clear_failure()
    second = await _runner(migration_api, source, failing_target).run()

    assert result_status(second) == "COMPLETE"
    for expected in pvc_fixture.datasets:
        await assert_dataset_matches(failing_target, expected)


@pytest.mark.asyncio
async def test_pvc_runner_copies_partial_payload_to_sqlite(
    migration_api,
    pvc_fixture_with_partial_payload,
    partial_payload,
    sqlite_target,
) -> None:
    """Partial KServe payloads survive the canonical transfer."""
    source = migration_api.PVCSource(
        pvc_fixture_with_partial_payload.directory,
        quiesced=True,
    )

    result = await _runner(migration_api, source, sqlite_target).run()

    assert result_status(result) == "COMPLETE"
    assert (
        await target_storage(sqlite_target).get_partial_payload(
            "pending-1",
            is_input=True,
            is_modelmesh=False,
        )
        == partial_payload
    )


@pytest.mark.asyncio
async def test_pvc_runner_copies_modelmesh_partial_payload_to_sqlite(
    migration_api,
    pvc_fixture_with_modelmesh_partial_payload,
    modelmesh_partial_payload,
    sqlite_target,
) -> None:
    """Partial ModelMesh payloads survive the canonical transfer."""
    source = migration_api.PVCSource(
        pvc_fixture_with_modelmesh_partial_payload.directory,
        quiesced=True,
    )

    result = await _runner(migration_api, source, sqlite_target).run()

    assert result_status(result) == "COMPLETE"
    assert (
        await target_storage(sqlite_target).get_partial_payload(
            "mm-1",
            is_input=True,
            is_modelmesh=True,
        )
        == modelmesh_partial_payload
    )


@pytest.mark.asyncio
async def test_pvc_runner_is_idempotent_and_verifies_sqlite_target(
    migration_api, pvc_fixture, sqlite_target
) -> None:
    """A completed migration can be rerun without duplicate rows."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    runner = _runner(migration_api, source, sqlite_target)

    first = await runner.run()
    second = await _runner(migration_api, source, sqlite_target).run()
    verification = await runner.verify()

    assert result_status(first) == "COMPLETE"
    assert result_status(second) == "COMPLETE"
    assert second.rows_written == 0
    assert verification.ok is True
    assert verification.rows_compared == 6


@pytest.mark.asyncio
async def test_pvc_runner_repairs_rows_after_verification_failure(
    migration_api, pvc_fixture, sqlite_target
) -> None:
    """A failed verification must make the next retry rewrite completed datasets."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    first = await _runner(migration_api, source, sqlite_target).run()
    storage = target_storage(sqlite_target)
    dataset = pvc_fixture.datasets[0]
    await storage.clear_dataset_rows(dataset.name)
    await storage.write_data(
        dataset.name,
        np.zeros((len(dataset.rows), len(dataset.column_names))),
        list(dataset.column_names),
    )

    second = await _runner(migration_api, source, sqlite_target).run()
    third = await _runner(migration_api, source, sqlite_target).run()

    assert result_status(first) == "COMPLETE"
    assert result_status(second) == "PARTIAL"
    assert result_status(third) == "COMPLETE"
    await assert_dataset_matches(sqlite_target, dataset)


@pytest.mark.asyncio
async def test_pvc_runner_rejects_changed_source_after_copy(
    migration_api, pvc_fixture, sqlite_target
) -> None:
    """A changed PVC cannot be declared safe after the original copy."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    runner = _runner(migration_api, source, sqlite_target)

    result = await runner.run()
    path = pvc_fixture.directory / f"{pvc_fixture.datasets[0].name}_trustyai.hdf5"
    with h5py.File(path, "r+") as h5_file:
        h5_file[pvc_fixture.datasets[0].name][0, 0] = 999.0

    verification = await runner.verify()

    assert result_status(result) == "COMPLETE"
    assert verification.ok is False
    assert verification.errors == ("source changed after preflight",)


@pytest.mark.asyncio
async def test_pvc_runner_verifies_partial_payloads(
    migration_api,
    pvc_fixture_with_partial_payload,
    sqlite_target,
) -> None:
    """A missing partial payload makes verification unsafe."""
    source = migration_api.PVCSource(
        pvc_fixture_with_partial_payload.directory,
        quiesced=True,
    )
    runner = _runner(migration_api, source, sqlite_target)

    result = await runner.run()
    await target_storage(sqlite_target).delete_partial_payload(
        "pending-1",
        is_input=True,
    )
    verification = await runner.verify()

    assert result_status(result) == "COMPLETE"
    assert verification.ok is False
    assert verification.partials_compared == 1
    assert verification.errors == ("missing target partial payload: pending-1",)


@pytest.mark.asyncio
async def test_pvc_runner_detects_changed_partial_payload(
    migration_api,
    pvc_fixture_with_partial_payload,
    partial_payload,
    sqlite_target,
) -> None:
    """Changing a partial payload changes the source snapshot digest."""
    source = migration_api.PVCSource(
        pvc_fixture_with_partial_payload.directory,
        quiesced=True,
    )
    runner = _runner(migration_api, source, sqlite_target)

    result = await runner.run()
    partial_path = pvc_fixture_with_partial_payload.directory / (
        f"{PARTIAL_INPUT_NAME}_trustyai.hdf5"
    )
    with h5py.File(partial_path, "r+") as h5_file:
        h5_file[PARTIAL_INPUT_NAME].attrs["pending-1"] = np.void(
            serialize_model(partial_payload.model_copy(update={"id": "changed"}))
        )

    verification = await runner.verify()

    assert result_status(result) == "COMPLETE"
    assert verification.ok is False
    assert verification.errors == ("source changed after preflight",)


@pytest.mark.asyncio
async def test_pvc_runner_rejects_reusing_namespace_for_new_source(
    migration_api, pvc_fixture, sqlite_target
) -> None:
    """A different source cannot append into an existing migration namespace."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    first = await _runner(migration_api, source, sqlite_target).run()
    path = pvc_fixture.directory / f"{pvc_fixture.datasets[0].name}_trustyai.hdf5"
    with h5py.File(path, "r+") as h5_file:
        h5_file[pvc_fixture.datasets[0].name][0, 0] = 999.0

    with pytest.raises(migration_api.MigrationPreflightError, match="namespace"):
        await _runner(migration_api, source, sqlite_target).run()

    assert result_status(first) == "COMPLETE"


@pytest.mark.asyncio
async def test_pvc_runner_reports_corrupt_serialized_rows(
    migration_api, corrupt_pvc_fixture, sqlite_target
) -> None:
    """Corrupt source rows produce an unsafe partial result."""
    source = migration_api.PVCSource(corrupt_pvc_fixture, quiesced=True)

    result = await _runner(migration_api, source, sqlite_target).run()

    assert result_status(result) in {"FAILED", "PARTIAL"}
    assert result.failed_datasets
    assert result.safe_to_remove_source is False
