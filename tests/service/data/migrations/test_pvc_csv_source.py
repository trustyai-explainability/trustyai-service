"""Tests for the CSV-on-PVC migration source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .conftest import result_status, target_storage
from .test_migration_contract import _runner

_TENSOR = "rO0ABXNyADNvcmcua2llLnRydXN0eWFpLmV4cGxhaW5hYmlsaXR5Lm1vZGVsLnRlbnNvci5UZW5zb3JFJ6gW0kbbrgIABUkACWRpbWVuc2lvbkkACG5FbnRyaWVzWwAEZGF0YXQAE1tMamF2YS9sYW5nL09iamVjdDtMAAhkYXRhdHlwZXQAEUxqYXZhL2xhbmcvQ2xhc3M7WwAKZGltZW5zaW9uc3QAAltJeHAAAAABAAAAAnVyABNbTGphdmEubGFuZy5Eb3VibGU74RKtiQCmVqYCAAB4cAAAAAJzcgAQamF2YS5sYW5nLkRvdWJsZYCzwkopa/sEAgABRAAFdmFsdWV4cgAQamF2YS5sYW5nLk51bWJlcoaslR0LlOCLAgAAeHA/9AAAAAAAAHNxAH4AB0AEAAAAAAAAdnEAfgAHdXIAAltJTbpgJnbqsqUCAAB4cAAAAAEAAAAC"  # pragma: allowlist secret


def _write_java_csv_model(directory: Path, model_id: str = "model-42") -> Path:
    """Write real Java-service filenames and sidecar field names."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "modelId": model_id,
        "inputTensorName": "instances",
        "outputTensorName": "predictions",
        "observations": 2,
        "recordedInferences": True,
        "inputSchema": {
            "items": {
                "text": {"type": "STRING", "name": "text", "columnIndex": 0},
                "embedding": {
                    "type": "TENSOR",
                    "name": "embedding",
                    "columnIndex": 1,
                },
                "active": {"type": "BOOL", "name": "active", "columnIndex": 2},
                "probability": {
                    "type": "FLOAT",
                    "name": "probability",
                    "columnIndex": 3,
                },
                "count": {"type": "INT32", "name": "count", "columnIndex": 4},
                "total": {"type": "INT64", "name": "total", "columnIndex": 5},
                "category": {"type": "MAP", "name": "category", "columnIndex": 6},
                "legacy": {"type": "UNKNOWN", "name": "legacy", "columnIndex": 7},
            },
            "nameMapping": {"text": "prompt"},
        },
        "outputSchema": {
            "items": {"score": {"type": "DOUBLE", "name": "score", "columnIndex": 8}},
            "nameMapping": {"score": "risk"},
        },
    }
    (directory / f"{model_id}-data.csv").write_text(
        f'"hello, world",{_TENSOR},true,0.12345679,2147483647,9223372036854775807,'
        '"{rank=1}",unknown,0.75\n'
        f"bye,{_TENSOR},false,0.5,-2147483648,-9223372036854775808,"
        '"{rank=2}",other,0.25\n',
        encoding="utf-8",
    )
    (directory / f"{model_id}-internal_data.csv").write_text(
        "organic,id-1,2024-02-03T04:05:06.123456789\nsynthetic,id-2,2024-02-03T04:06:07\n",
        encoding="utf-8",
    )
    metadata_path = directory / f"{model_id}-metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


@pytest.fixture
def java_csv_pvc(tmp_path: Path) -> Path:
    """Provide one Java-format model with aliases, tensors, and row metadata."""
    _write_java_csv_model(tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_pvc_csv_source_discovers_and_decodes_java_storage(
    migration_api, java_csv_pvc: Path
) -> None:
    """Map sidecars, Java tensors, aliases, and internal rows to datasets."""
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)

    manifest = await source.preflight()

    assert manifest.source_type == "PVC_CSV"
    assert manifest.total_files == 3
    assert manifest.total_rows == 6
    descriptors = {dataset.name: dataset for dataset in manifest.datasets}
    inputs = descriptors["model-42_inputs"]
    outputs = descriptors["model-42_outputs"]
    metadata = descriptors["model-42_metadata"]
    assert inputs.column_names == (
        "text",
        "embedding",
        "active",
        "probability",
        "count",
        "total",
        "category",
        "legacy",
    )
    assert inputs.aliases == (
        "prompt",
        "embedding",
        "active",
        "probability",
        "count",
        "total",
        "category",
        "legacy",
    )
    assert inputs.column_types == (
        "STRING",
        "TENSOR",
        "BOOL",
        "FLOAT",
        "INT32",
        "INT64",
        "MAP",
        "UNKNOWN",
    )
    assert inputs.service_metadata == {
        "input_tensor_name": "instances",
        "output_tensor_name": "predictions",
        "recorded_inferences": True,
    }
    assert outputs.column_names == ("score",)
    assert outputs.aliases == ("risk",)
    assert outputs.column_types == ("DOUBLE",)
    assert metadata.column_names == ("timestamp", "rowId", "tag")

    input_rows = await source.read_rows(inputs, 0, 2)
    assert input_rows.tolist() == [
        [
            "hello, world",
            [1.25, 2.5],
            True,
            float(np.float32(0.12345679)),
            2147483647,
            9223372036854775807,
            "{rank=1}",
            "unknown",
        ],
        [
            "bye",
            [1.25, 2.5],
            False,
            0.5,
            -2147483648,
            -9223372036854775808,
            "{rank=2}",
            "other",
        ],
    ]
    output_rows = await source.read_rows(outputs, 0, 2)
    np.testing.assert_allclose(output_rows.astype(float), [[0.75], [0.25]])
    metadata_rows = await source.read_rows(metadata, 0, 2)
    assert metadata_rows[0].tolist() == [
        "2024-02-03T04:05:06.123456789",
        "id-1",
        "organic",
    ]


@pytest.mark.asyncio
async def test_pvc_csv_source_reads_tensor_only_rows(
    migration_api, tmp_path: Path
) -> None:
    """Tensor-only rows must not be coerced into a multidimensional row array."""
    model_id = "tensor-model"
    (tmp_path / f"{model_id}-data.csv").write_text(
        f"{_TENSOR},0.75\n{_TENSOR},0.25\n",
        encoding="utf-8",
    )
    (tmp_path / f"{model_id}-internal_data.csv").write_text(
        "organic,id-1,2024-02-03T04:05:06\nsynthetic,id-2,2024-02-03T04:06:07\n",
        encoding="utf-8",
    )
    (tmp_path / f"{model_id}-metadata.json").write_text(
        json.dumps(
            {
                "modelId": model_id,
                "inputTensorName": "instances",
                "outputTensorName": "predictions",
                "observations": 2,
                "recordedInferences": True,
                "inputSchema": {
                    "items": {
                        "embedding": {
                            "type": "TENSOR",
                            "name": "embedding",
                            "columnIndex": 0,
                        }
                    }
                },
                "outputSchema": {
                    "items": {
                        "score": {
                            "type": "DOUBLE",
                            "name": "score",
                            "columnIndex": 1,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    source = migration_api.PVCCSVSource(tmp_path, quiesced=True)
    manifest = await source.preflight()
    inputs = next(item for item in manifest.datasets if item.name.endswith("_inputs"))

    rows = await source.read_rows(inputs, 0, 2)

    assert rows.tolist() == [[[1.25, 2.5]], [[1.25, 2.5]]]


@pytest.mark.asyncio
async def test_pvc_csv_source_accepts_java_legacy_schema_index(
    migration_api, java_csv_pvc: Path
) -> None:
    """Java metadata may serialize SchemaItem's index as ``index``."""
    metadata_path = java_csv_pvc / "model-42-metadata.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    for schema_name in ("inputSchema", "outputSchema"):
        schema = document[schema_name]
        for item in schema["items"].values():
            item["index"] = item.pop("columnIndex")
        schema["remapCount"] = 0
        schema["nameMappedItems"] = schema["items"]
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)

    manifest = await source.preflight()
    descriptors = {dataset.name: dataset for dataset in manifest.datasets}

    assert descriptors["model-42_inputs"].row_count == 2
    rows = await source.read_rows(descriptors["model-42_outputs"], 0, 2)
    np.testing.assert_allclose(rows.astype(float), [[0.75], [0.25]])


@pytest.mark.asyncio
async def test_pvc_csv_source_allows_alias_reuse_across_input_and_output(
    migration_api, java_csv_pvc: Path
) -> None:
    """Input and output aliases occupy separate logical dataset schemas."""
    metadata_path = java_csv_pvc / "model-42-metadata.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["inputSchema"]["nameMapping"]["text"] = "risk"
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)

    manifest = await source.preflight()
    descriptors = {dataset.name: dataset for dataset in manifest.datasets}

    assert descriptors["model-42_inputs"].aliases[0] == "risk"
    assert descriptors["model-42_outputs"].aliases == ("risk",)


@pytest.mark.asyncio
async def test_pvc_csv_source_rejects_mismatched_data_and_sidecar_counts(
    migration_api, java_csv_pvc: Path
) -> None:
    """A partial sidecar must fail before the runner writes any target rows."""
    internal_path = java_csv_pvc / "model-42-internal_data.csv"
    internal_path.write_text(
        "organic,id-1,2024-02-03T04:05:06.123456789\n", encoding="utf-8"
    )
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)

    with pytest.raises(migration_api.MigrationValidationError, match="row count"):
        await source.preflight()


@pytest.mark.asyncio
async def test_pvc_csv_source_rejects_malformed_schema_before_transfer(
    migration_api, java_csv_pvc: Path
) -> None:
    """Invalid global column positions cannot be guessed or silently reordered."""
    metadata_path = java_csv_pvc / "model-42-metadata.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["inputSchema"]["items"]["text"]["columnIndex"] = 9
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)

    with pytest.raises(migration_api.MigrationValidationError, match="column index"):
        await source.preflight()


@pytest.mark.asyncio
async def test_pvc_csv_source_reads_bounded_offset_batches(
    migration_api, java_csv_pvc: Path
) -> None:
    """Resume reads seek to the requested logical row without loading prior rows."""
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)
    manifest = await source.preflight()
    output = next(item for item in manifest.datasets if item.name == "model-42_outputs")

    rows = await source.read_rows(output, 1, 1)

    assert rows.tolist() == [[0.25]]


@pytest.mark.asyncio
async def test_pvc_csv_source_preserves_java_non_finite_floats(
    migration_api, java_csv_pvc: Path, sqlite_target
) -> None:
    """Java Float/Double/Tensor values include IEEE non-finite values."""
    import base64

    import javaobj

    tensor = javaobj.loads(base64.b64decode(_TENSOR))
    tensor.data[0].value = float("nan")
    encoded_tensor = base64.b64encode(javaobj.dumps(tensor)).decode("ascii")
    data_path = java_csv_pvc / "model-42-data.csv"
    data_path.write_text(
        f'"hello, world",{encoded_tensor},true,NaN,1,1,"{{rank=1}}",unknown,Infinity\n'
        f'bye,{encoded_tensor},false,-Infinity,1,1,"{{rank=2}}",other,-Infinity\n',
        encoding="utf-8",
    )
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)
    manifest = await source.preflight()
    descriptors = {dataset.name: dataset for dataset in manifest.datasets}

    inputs = await source.read_rows(descriptors["model-42_inputs"], 0, 2)
    outputs = await source.read_rows(descriptors["model-42_outputs"], 0, 2)

    assert np.isnan(inputs[0, 3])
    assert np.isnan(inputs[0, 1][0])
    assert np.isneginf(inputs[1, 3])
    assert np.isposinf(outputs[0, 0])
    assert np.isneginf(outputs[1, 0])
    result = await _runner(migration_api, source, sqlite_target).run()
    assert result_status(result) == "COMPLETE"
    assert result.safe_to_remove_source is True


@pytest.mark.asyncio
async def test_pvc_csv_source_accepts_fields_above_python_default_limit(
    migration_api, java_csv_pvc: Path
) -> None:
    """Large strings and serialized tensors are not truncated by csv's default cap."""
    import csv

    original_limit = csv.field_size_limit()
    long_text = "x" * 150_000
    data_path = java_csv_pvc / "model-42-data.csv"
    data_path.write_text(
        f'"{long_text}",{_TENSOR},true,0.5,1,1,"{{rank=1}}",unknown,0.75\n'
        f'bye,{_TENSOR},false,0.5,1,1,"{{rank=2}}",other,0.25\n',
        encoding="utf-8",
    )
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)
    manifest = await source.preflight()
    inputs = next(item for item in manifest.datasets if item.name == "model-42_inputs")

    rows = await source.read_rows(inputs, 0, 1)

    assert len(rows[0, 0]) == len(long_text)
    assert csv.field_size_limit() == original_limit


@pytest.mark.asyncio
async def test_csv_field_limit_is_bounded_and_restored(
    migration_api, java_csv_pvc, monkeypatch
):
    """Oversized cells fail and even a failed parser restores the global limit."""
    import csv

    from trustyai_service.service.data.migrations import pvc_csv

    original_limit = csv.field_size_limit()
    monkeypatch.setattr(pvc_csv, "_MAX_CSV_FIELD_SIZE", 16)
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)
    assert csv.field_size_limit() == original_limit
    with pytest.raises(migration_api.MigrationValidationError, match="field larger"):
        await source.preflight()
    assert csv.field_size_limit() == original_limit


@pytest.mark.asyncio
async def test_csv_source_rejects_unselected_models(migration_api, java_csv_pvc):
    """Other filename generations cannot be omitted while claiming completion."""
    (java_csv_pvc / "other-old.csv").write_text("unselected", encoding="utf-8")
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)
    with pytest.raises(
        migration_api.MigrationValidationError, match="unselected storage"
    ):
        await source.preflight()


@pytest.mark.asyncio
async def test_pvc_csv_source_rejects_observation_count_mismatch(
    migration_api, java_csv_pvc: Path
) -> None:
    """Sidecar observations must agree with the actual paired CSV records."""
    metadata_path = java_csv_pvc / "model-42-metadata.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["observations"] = 99
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)

    with pytest.raises(
        migration_api.MigrationValidationError, match="observation count"
    ):
        await source.preflight()


@pytest.mark.asyncio
async def test_pvc_csv_source_rejects_invalid_typed_cells(
    migration_api, java_csv_pvc: Path
) -> None:
    """Malformed typed values are found before any transfer begins."""
    data_path = java_csv_pvc / "model-42-data.csv"
    data_path.write_text(
        f'"hello, world",{_TENSOR},true,0.5,1,1,"{{rank=1}}",unknown,not-a-number\n'
        f'bye,{_TENSOR},false,0.5,1,1,"{{rank=2}}",other,0.25\n',
        encoding="utf-8",
    )
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)

    with pytest.raises(migration_api.MigrationValidationError, match="DOUBLE"):
        await source.preflight()


@pytest.mark.asyncio
async def test_pvc_csv_source_rejects_malformed_tensor_payload(
    migration_api, java_csv_pvc: Path
) -> None:
    """Malformed Java object-stream cells never reach the database target."""
    data_path = java_csv_pvc / "model-42-data.csv"
    data_path.write_text(
        "hello,not-base64,true,0.5,1,1,{rank=1},unknown,0.75\n"
        f'bye,{_TENSOR},false,0.5,1,1,"{{rank=2}}",other,0.25\n',
        encoding="utf-8",
    )
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)

    with pytest.raises(
        migration_api.MigrationValidationError, match="serialized Tensor"
    ):
        await source.preflight()


@pytest.mark.asyncio
async def test_pvc_csv_source_rejects_missing_or_linked_sidecars(
    migration_api, java_csv_pvc: Path, tmp_path: Path
) -> None:
    """Every discovered model must have regular, in-directory sidecar files."""
    (java_csv_pvc / "model-42-internal_data.csv").unlink()
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)
    with pytest.raises(migration_api.MigrationValidationError, match="missing"):
        await source.preflight()

    other_directory = tmp_path / "linked"
    _write_java_csv_model(other_directory)
    link = other_directory / "model-42-internal_data.csv"
    target = other_directory / "internal-copy.csv"
    link.rename(target)
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")
    linked_source = migration_api.PVCCSVSource(other_directory, quiesced=True)
    with pytest.raises(migration_api.MigrationValidationError, match="symlink"):
        await linked_source.preflight()


@pytest.mark.asyncio
async def test_pvc_csv_source_rejects_path_escape_in_descriptor(
    migration_api, java_csv_pvc: Path
) -> None:
    """A caller cannot substitute a discovered descriptor's source path."""
    from dataclasses import replace

    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)
    manifest = await source.preflight()
    dataset = next(item for item in manifest.datasets if item.name == "model-42_inputs")

    with pytest.raises(
        migration_api.MigrationValidationError, match="unexpected source"
    ):
        await source.read_rows(replace(dataset, source_file="../outside.csv"), 0, 1)


@pytest.mark.asyncio
async def test_pvc_csv_to_sqlite_preserves_data_and_java_model_metadata(
    migration_api, java_csv_pvc: Path, sqlite_target
) -> None:
    """SQLite receives logical rows, aliases, types, and Java service metadata."""
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)
    result = await _runner(migration_api, source, sqlite_target, batch_size=1).run()

    assert result_status(result) == "COMPLETE"
    repeated_result = await _runner(
        migration_api,
        migration_api.PVCCSVSource(java_csv_pvc, quiesced=True),
        sqlite_target,
        batch_size=1,
    ).run()
    assert result_status(repeated_result) == "COMPLETE"
    assert repeated_result.rows_written == 0
    storage = target_storage(sqlite_target)
    assert await storage.list_all_datasets() == [
        "model-42_inputs",
        "model-42_metadata",
        "model-42_outputs",
    ]
    assert await storage.get_aliased_column_names("model-42_inputs") == [
        "prompt",
        "embedding",
        "active",
        "probability",
        "count",
        "total",
        "category",
        "legacy",
    ]
    assert await storage.get_column_types("model-42_inputs") == [
        "STRING",
        "TENSOR",
        "BOOL",
        "FLOAT",
        "INT32",
        "INT64",
        "MAP",
        "UNKNOWN",
    ]
    np.testing.assert_allclose(
        (await storage.read_data("model-42_outputs", 0, 2)).astype(float),
        [[0.75], [0.25]],
    )
    metadata = await storage.get_service_metadata("model-42_inputs")
    assert metadata == {
        "input_tensor_name": "instances",
        "output_tensor_name": "predictions",
        "recorded_inferences": True,
    }


@pytest.mark.asyncio
async def test_csv_batches_do_not_rescan_multiline_records(
    migration_api, java_csv_pvc, monkeypatch
) -> None:
    """Sequential batches parse each record once, including quoted newlines."""
    import csv

    path = java_csv_pvc / "model-42-data.csv"
    path.write_text(path.read_text().replace("hello, world", "hello,\nworld"))
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)
    manifest = await source.preflight()
    dataset = next(item for item in manifest.datasets if item.name.endswith("_inputs"))
    real_reader = csv.reader
    records_read = 0

    def counting_reader(*args, **kwargs):
        nonlocal records_read
        for row in real_reader(*args, **kwargs):
            records_read += 1
            yield row

    monkeypatch.setattr(csv, "reader", counting_reader)
    first = await source.read_rows(dataset, 0, 1)
    second = await source.read_rows(dataset, 1, 1)
    assert first[0, 0] == "hello,\nworld"
    assert second[0, 0] == "bye"
    assert records_read == 2
    # Nonsequential reads and a new preflight must still restart correctly.
    assert (await source.read_rows(dataset, 0, 1))[0, 0] == "hello,\nworld"
    await source.preflight()
    assert (await source.read_rows(dataset, 1, 1))[0, 0] == "bye"


@pytest.mark.asyncio
async def test_csv_migration_is_readable_by_production_service(
    migration_api, java_csv_pvc, tmp_path, monkeypatch
) -> None:
    """The actual SQL reader and metadata API restore the imported contract."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from trustyai_service.service.data.datasources.data_source import DataSource
    from trustyai_service.service.data.storage import GlobalStorageInterface
    from trustyai_service.service.data.storage.sqlite.sqlite import SQLiteStorage

    path = tmp_path / "cutover.sqlite"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    target = migration_api.DatabaseTarget(engine)
    try:
        result = await _runner(
            migration_api,
            migration_api.PVCCSVSource(java_csv_pvc, quiesced=True),
            target,
        ).run()
        assert result_status(result) == "COMPLETE"
    finally:
        await engine.dispose()
    storage = SQLiteStorage(str(path))
    monkeypatch.setattr(GlobalStorageInterface, "_instance", storage)
    try:
        assert (await storage.read_data("model-42_outputs")).tolist() == [
            [0.75],
            [0.25],
        ]
        metadata = await DataSource().get_metadata("model-42")
        assert metadata.observations == 2
        assert metadata.input_tensor_name == "instances"
        assert metadata.output_tensor_name == "predictions"
        assert metadata.recorded_inferences is True
        assert metadata.input_schema.items["embedding"].type == "TENSOR"
        assert metadata.input_schema.items["active"].type == "BOOL"
        assert metadata.input_schema.items["prompt"].type == "STRING"
        assert metadata.output_schema.items["risk"].type == "DOUBLE"
        assert await storage.get_original_column_names("model-42_outputs") == ["score"]
        assert await storage.get_aliased_column_names("model-42_outputs") == ["risk"]
    finally:
        storage._engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tensor_only", [False, True], ids=["mixed-cells", "tensor-only"]
)
async def test_csv_to_mariadb_to_postgres_preserves_public_metadata(
    migration_api, java_csv_pvc, target_factory, monkeypatch, tensor_only
) -> None:
    """Both required live servers exercise the CSV two-hop cutover contract."""
    from trustyai_service.service.data.datasources.data_source import DataSource
    from trustyai_service.service.data.storage import GlobalStorageInterface
    from trustyai_service.service.data.storage.postgres.postgres import (
        PostgreSQLStorage,
    )

    if tensor_only:
        metadata_path = java_csv_pvc / "model-42-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["inputSchema"] = {
            "items": {
                "embedding": {"type": "TENSOR", "name": "embedding", "columnIndex": 0}
            },
            "nameMapping": {},
        }
        metadata["outputSchema"]["items"]["score"]["columnIndex"] = 1
        metadata_path.write_text(json.dumps(metadata))
        (java_csv_pvc / "model-42-data.csv").write_text(
            f"{_TENSOR},0.75\n{_TENSOR},0.25\n"
        )

    maria = await target_factory("mariadb")
    postgres = await target_factory("postgres")
    first = await _runner(
        migration_api, migration_api.PVCCSVSource(java_csv_pvc, quiesced=True), maria
    ).run()
    assert result_status(first) == "COMPLETE"
    second = await _runner(
        migration_api,
        migration_api.MariaDBSource(maria.engine, schema="trustyai_v2", quiesced=True),
        postgres,
    ).run()
    assert result_status(second) == "COMPLETE"
    if tensor_only:
        rows = await target_storage(postgres).read_data("model-42_inputs")
        assert rows.shape == (2, 1)
        assert rows.tolist() == [[[1.25, 2.5]], [[1.25, 2.5]]]
    url = postgres.engine.url
    reader = PostgreSQLStorage(
        user=url.username or "",
        password=url.password or "",
        host=url.host or "",
        port=url.port or 5432,
        database=url.database or "",
    )
    monkeypatch.setattr(GlobalStorageInterface, "_instance", reader)
    try:
        metadata = await DataSource().get_metadata("model-42")
        assert metadata.input_tensor_name == "instances"
        assert metadata.output_tensor_name == "predictions"
        assert metadata.recorded_inferences is True
        assert metadata.input_schema.items["embedding"].type == "TENSOR"
        assert metadata.observations == 2
        assert (await reader.read_data("model-42_outputs")).tolist() == [[0.75], [0.25]]
    finally:
        reader._engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["mariadb", "postgres"])
async def test_pvc_csv_to_live_database_targets(
    migration_api, java_csv_pvc: Path, target_factory, target_kind: str
) -> None:
    """The same Java source contract is exercised on both supported SQL dialects."""
    target = await target_factory(target_kind)
    source = migration_api.PVCCSVSource(java_csv_pvc, quiesced=True)

    result = await _runner(migration_api, source, target, batch_size=1).run()

    assert result_status(result) == "COMPLETE"
    storage = target_storage(target)
    assert await storage.dataset_rows("model-42_metadata") == 2
    assert await storage.get_aliased_column_names("model-42_outputs") == ["risk"]
