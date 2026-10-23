"""Local runner coverage using SQLite as a dialect-neutral test target."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from trustyai_service.service.data.storage.pvc import PARTIAL_INPUT_NAME
from trustyai_service.service.serialization import serialize_model

from .conftest import (
    assert_dataset_matches,
    create_test_target,
    result_status,
    target_storage,
)
from .test_migration_contract import _runner


@pytest.mark.parametrize(
    ("expected", "actual", "equal"),
    [
        (np.array([float("nan"), 1]), np.array([float("nan"), 1]), True),
        (np.array(["x", float("nan")], dtype=object), ["x", float("nan")], True),
        (np.array([float("nan")]), np.array([0]), False),
        (np.array([np.float32("nan")], dtype=object), [float("nan")], True),
        (np.empty((0, 2)), np.empty((1, 0)), False),
    ],
)
def test_tensor_comparison_handles_nan_and_mixed_values(expected, actual, equal):
    """Verification handles NaNs without applying isnan to string objects."""
    from trustyai_service.service.data.migrations.runner import _values_equal

    assert _values_equal(expected, actual) is equal


@pytest.mark.asyncio
async def test_checkpoint_outage_returns_failure_instead_of_escaping(
    migration_api, pvc_fixture, sqlite_target, monkeypatch
):
    """Failure recording must never mask the transfer error or imply success."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)

    async def unavailable(*_args, **_kwargs):
        msg = "database unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(sqlite_target, "write_batch", unavailable)
    monkeypatch.setattr(sqlite_target, "save_migration_dataset_state", unavailable)
    result = await _runner(migration_api, source, sqlite_target).run()
    assert result_status(result) == "FAILED"
    assert result.safe_to_remove_source is False
    assert len(result.failed_datasets) == len(pvc_fixture.datasets)
    assert all("failure checkpoint" in item.error for item in result.failed_datasets)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on_call", [1, 2])
async def test_run_checkpoint_outage_fails_closed(
    migration_api, pvc_fixture, sqlite_target, monkeypatch, fail_on_call
):
    """Neither an initial nor a final checkpoint failure permits source removal."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    save = sqlite_target.save_migration_run
    calls = 0

    async def unavailable(values):
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            msg = "database unavailable"
            raise RuntimeError(msg)
        await save(values)

    monkeypatch.setattr(sqlite_target, "save_migration_run", unavailable)
    result = await _runner(migration_api, source, sqlite_target).run()
    assert result_status(result) == "FAILED"
    assert result.safe_to_remove_source is False
    assert "checkpoint" in result.error
    if fail_on_call == 1:
        assert await sqlite_target.list_target_datasets() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("modelmesh", [False, True])
async def test_output_partial_payloads_survive_migration(
    migration_api, pvc_fixture, sqlite_target, modelmesh, modelmesh_partial_payload
):
    """Both output-side formats are discovered, transferred and verified."""
    from trustyai_service.endpoints.consumer import KServeInferenceResponse
    from trustyai_service.service.data.storage.pvc import PARTIAL_OUTPUT_NAME

    payload = (
        modelmesh_partial_payload
        if modelmesh
        else KServeInferenceResponse(id="out-1", model_name="model", outputs=[])
    )
    path = pvc_fixture.directory / f"{PARTIAL_OUTPUT_NAME}_trustyai.hdf5"
    with h5py.File(path, "w") as source_file:
        dataset = source_file.create_dataset(PARTIAL_OUTPUT_NAME, dtype="f")
        dataset.attrs["out-1"] = np.void(serialize_model(payload))
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    manifest = await source.preflight()
    assert manifest.partial_payloads[0].is_input is False
    result = await _runner(migration_api, source, sqlite_target).run()
    assert result_status(result) == "COMPLETE"
    assert (
        await sqlite_target.storage.get_partial_payload(
            "out-1", is_input=False, is_modelmesh=modelmesh
        )
        == payload
    )


@pytest.mark.asyncio
async def test_dry_run_detects_corrupt_rows_without_writes(
    migration_api, corrupt_pvc_fixture, sqlite_target
):
    """Dry-run must decode all rows, not only inspect metadata."""
    source = migration_api.PVCSource(corrupt_pvc_fixture, quiesced=True)
    with pytest.raises(migration_api.MigrationValidationError):
        await _runner(migration_api, source, sqlite_target, dry_run=True).run()
    assert await sqlite_target.list_target_datasets() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("checksums", [True, False])
async def test_no_checksums_skips_only_rehash_not_content_verification(
    migration_api, pvc_fixture, sqlite_target, monkeypatch, checksums
):
    """Disabling the rehash does not disable destination fidelity checks."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    preflight = source.preflight
    calls = 0

    async def counted_preflight():
        nonlocal calls
        calls += 1
        return await preflight()

    monkeypatch.setattr(source, "preflight", counted_preflight)
    runner = _runner(migration_api, source, sqlite_target, verify_checksums=checksums)
    result = await runner.run()
    assert result_status(result) == "COMPLETE"
    assert calls == 1 + int(checksums)
    await sqlite_target.storage.clear_dataset_rows(pvc_fixture.datasets[0].name)
    verification = await runner.verify()
    assert verification.ok is False
    assert any("row-count mismatch" in error for error in verification.errors)
    assert calls == 1 + 2 * int(checksums)


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
    failing_target = create_test_target(
        migration_api,
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
async def test_completed_migration_never_replays_divergent_target(
    migration_api, pvc_fixture, sqlite_target
) -> None:
    """Completed cutovers cannot authorize replacement of subsequently edited data."""
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

    assert result_status(first) == "COMPLETE"
    for resume in (True, False):
        with pytest.raises(migration_api.MigrationPreflightError, match="completed"):
            await _runner(migration_api, source, sqlite_target, resume=resume).run()
    np.testing.assert_equal(
        await storage.read_data(dataset.name),
        np.zeros((len(dataset.rows), len(dataset.column_names))),
    )


@pytest.mark.asyncio
async def test_completed_migration_preserves_appended_target_rows(
    migration_api, pvc_fixture, sqlite_target
) -> None:
    """Rerunning after cutover must not erase newly ingested observations."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    await _runner(migration_api, source, sqlite_target).run()
    dataset = pvc_fixture.datasets[0]
    storage = target_storage(sqlite_target)
    await storage.write_data(
        dataset.name,
        np.asarray([[10.0, 11.0, 12.0]]),
        list(dataset.column_names),
    )
    before = await storage.read_data(dataset.name)
    with pytest.raises(migration_api.MigrationPreflightError, match="completed"):
        await _runner(migration_api, source, sqlite_target).run()
    np.testing.assert_equal(await storage.read_data(dataset.name), before)


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

    with pytest.raises(migration_api.MigrationPreflightError, match="target database"):
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
