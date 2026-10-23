"""Unit tests for PVC source discovery and decoding."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_pvc_source_rejects_unselected_storage_files(migration_api, pvc_fixture):
    """A historical filename must not be silently omitted from a cutover."""
    (pvc_fixture.directory / "other_old.hdf5").write_bytes(b"unselected")
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)
    with pytest.raises(
        migration_api.MigrationValidationError, match="unselected storage"
    ):
        await source.preflight()


@pytest.mark.asyncio
async def test_pvc_source_builds_manifest_and_reads_logical_batches(
    migration_api, pvc_fixture
) -> None:
    """PVC metadata and rows are exposed in the canonical logical form."""
    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)

    manifest = await source.preflight()

    assert manifest.source_type == "PVC"
    assert manifest.total_files == 3
    assert [dataset.name for dataset in manifest.datasets] == sorted(
        dataset.name for dataset in pvc_fixture.datasets
    )
    assert manifest.total_rows == 6
    for expected, descriptor in zip(
        sorted(pvc_fixture.datasets, key=lambda dataset: dataset.name),
        manifest.datasets,
        strict=True,
    ):
        assert descriptor.shape == (len(expected.column_names),)
        rows = await source.read_rows(descriptor, 0, descriptor.row_count)
        assert [tuple(row.tolist()) for row in rows] == [
            tuple(row) for row in expected.rows
        ]


@pytest.mark.asyncio
async def test_pvc_source_decodes_partial_payload_file(
    migration_api, pvc_fixture_with_partial_payload, partial_payload
) -> None:
    """Persisted KServe partial input payloads are part of the source snapshot."""
    source = migration_api.PVCSource(
        pvc_fixture_with_partial_payload.directory,
        quiesced=True,
    )

    manifest = await source.preflight()

    assert len(manifest.partial_payloads) == 1
    assert manifest.partial_payloads[0].payload_id == "pending-1"
    assert manifest.partial_payloads[0].payload == partial_payload


@pytest.mark.asyncio
async def test_pvc_source_classifies_modelmesh_partial_payloads(
    migration_api,
    pvc_fixture_with_modelmesh_partial_payload,
    modelmesh_partial_payload,
) -> None:
    """ModelMesh partial payloads are detected instead of failing preflight."""
    source = migration_api.PVCSource(
        pvc_fixture_with_modelmesh_partial_payload.directory,
        quiesced=True,
    )

    manifest = await source.preflight()

    by_id = {record.payload_id: record for record in manifest.partial_payloads}
    assert by_id["pending-1"].is_modelmesh is False
    assert by_id["mm-1"].is_modelmesh is True
    assert by_id["mm-1"].payload == modelmesh_partial_payload


@pytest.mark.asyncio
async def test_pvc_source_rejects_missing_column_metadata(
    migration_api, invalid_pvc_fixture
) -> None:
    """A source without column metadata cannot be migrated safely."""
    source = migration_api.PVCSource(invalid_pvc_fixture, quiesced=True)

    with pytest.raises(migration_api.MigrationValidationError, match="column"):
        await source.preflight()


@pytest.mark.asyncio
async def test_pvc_source_rejects_matching_symlinks(migration_api, pvc_fixture) -> None:
    """A matching symlink cannot make the source read outside its directory."""
    link = pvc_fixture.directory / "linked_trustyai.hdf5"
    try:
        link.symlink_to(pvc_fixture.directory / "missing.hdf5")
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    source = migration_api.PVCSource(pvc_fixture.directory, quiesced=True)

    with pytest.raises(migration_api.MigrationValidationError, match="symlink"):
        await source.preflight()
