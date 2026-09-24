"""Tests for bounded local-explanation storage loading."""

from __future__ import annotations

from dataclasses import fields
from inspect import signature
from typing import Any, cast

import numpy as np
import pytest

from trustyai_service.service.constants import (
    INPUT_SUFFIX,
    METADATA_SUFFIX,
    OUTPUT_SUFFIX,
    SYNTHETIC_TAG,
)
from trustyai_service.service.data.local_explanation import (
    LocalExplanationData,
    LocalExplanationMissingOutputError,
    load_local_explanation_data,
)


class _Storage:
    """Small async storage double that exposes row-level read observations."""

    def __init__(
        self,
        metadata: np.ndarray,
        inputs: np.ndarray,
        outputs: np.ndarray | None = None,
        **options: object,
    ) -> None:
        self.metadata = metadata
        self.inputs = inputs
        self.outputs = outputs
        self.input_names = cast("list[str] | None", options.get("input_names")) or [
            "feature-a",
            "feature-b",
        ]
        self.output_names = cast("list[str] | None", options.get("output_names")) or [
            "score"
        ]
        self.tensor_names = cast("dict[str, str] | None", options.get("tensor_names"))
        self.fail_get_metadata = bool(options.get("fail_get_metadata", False))
        self.metadata_accesses = 0
        self.collapse_single_row = bool(options.get("collapse_single_row", False))
        self.metadata_rows = cast("int | None", options.get("metadata_rows"))
        self.input_rows = cast("int | None", options.get("input_rows"))
        self.output_rows = cast("int | None", options.get("output_rows"))
        self.reads: list[tuple[str, int, int | None]] = []

    async def dataset_exists(self, dataset_name: str) -> bool:
        if dataset_name.endswith(OUTPUT_SUFFIX):
            return self.outputs is not None
        return True

    async def dataset_rows(self, dataset_name: str) -> int:
        if dataset_name.endswith(METADATA_SUFFIX) and self.metadata_rows is not None:
            return self.metadata_rows
        if dataset_name.endswith(INPUT_SUFFIX) and self.input_rows is not None:
            return self.input_rows
        if dataset_name.endswith(OUTPUT_SUFFIX) and self.output_rows is not None:
            return self.output_rows
        if dataset_name.endswith(METADATA_SUFFIX):
            return len(self.metadata)
        if dataset_name.endswith(OUTPUT_SUFFIX):
            return 0 if self.outputs is None else len(self.outputs)
        return len(self.inputs)

    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        if dataset_name.endswith(INPUT_SUFFIX):
            return list(self.input_names)
        return list(self.output_names)

    async def get_metadata(self, _model_id: str) -> dict[str, str] | None:
        self.metadata_accesses += 1
        if self.fail_get_metadata:
            msg = "MODEL loading must not inspect aggregate metadata"
            raise AssertionError(msg)
        return self.tensor_names

    async def read_data(
        self,
        dataset_name: str,
        start_row: int = 0,
        n_rows: int | None = None,
    ) -> np.ndarray:
        self.reads.append((dataset_name, start_row, n_rows))
        if dataset_name.endswith(METADATA_SUFFIX):
            rows: Any = self.metadata
        elif dataset_name.endswith(INPUT_SUFFIX):
            rows = self.inputs
        else:
            if self.outputs is None:
                msg = "the loader attempted to read missing outputs"
                raise AssertionError(msg)
            rows = self.outputs
        end = None if n_rows is None else start_row + n_rows
        result = rows[start_row:end]
        if self.collapse_single_row and len(result) == 1:
            return result[0]
        return result


class _VirtualStorage:
    """Lazy storage double for scans that cross the former row-count cap."""

    def __init__(
        self,
        *,
        target_indices: set[int],
        synthetic_indices: set[int] | None = None,
        organic_indices: set[int] | None = None,
    ) -> None:
        self.row_count = 1_000_001
        self.target_indices = target_indices
        self.synthetic_indices = synthetic_indices or set()
        self.organic_indices = organic_indices

    async def dataset_exists(self, dataset_name: str) -> bool:
        return not dataset_name.endswith(OUTPUT_SUFFIX)

    async def dataset_rows(self, _dataset_name: str) -> int:
        return self.row_count

    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        return ["feature-a", "feature-b"] if dataset_name.endswith(INPUT_SUFFIX) else []

    async def get_metadata(self, _model_id: str) -> None:
        return None

    async def read_data(
        self,
        dataset_name: str,
        start_row: int = 0,
        n_rows: int | None = None,
    ) -> np.ndarray:
        count = self.row_count - start_row if n_rows is None else n_rows
        count = min(count, self.row_count - start_row)
        indices = range(start_row, start_row + count)
        if dataset_name.endswith(METADATA_SUFFIX):
            rows = np.empty((count, 4), dtype=object)
            for row_offset, index in enumerate(indices):
                row_id = b"target" if index in self.target_indices else f"row-{index}"
                if self.organic_indices is None:
                    synthetic = index in self.synthetic_indices
                else:
                    synthetic = index not in self.organic_indices
                rows[row_offset] = [
                    row_id,
                    "2026-09-22T00:00:00",
                    0.0,
                    [SYNTHETIC_TAG] if synthetic else [],
                ]
            return rows
        values = np.arange(start_row, start_row + count, dtype=float)
        return np.column_stack((values, values + 100.0))


def _metadata(
    ids: list[object],
    tags: list[object] | None = None,
) -> np.ndarray:
    row_tags = tags or [[] for _ in ids]
    return np.array(
        [
            [row_id, "2026-09-22T00:00:00", 0.0, row_tag]
            for row_id, row_tag in zip(ids, row_tags, strict=True)
        ],
        dtype=object,
    )


def _storage(
    ids: list[object],
    *,
    tags: list[object] | None = None,
    inputs: np.ndarray | None = None,
    outputs: np.ndarray | None = None,
    **options: object,
) -> _Storage:
    if inputs is None:
        inputs = np.column_stack(
            (
                np.arange(len(ids), dtype=float),
                np.arange(len(ids), dtype=float) + 100.0,
            )
        )
    return _Storage(_metadata(ids, tags), inputs, outputs, **options)


@pytest.mark.asyncio
async def test_loader_has_only_the_canonical_public_interface() -> None:
    """Expose the exact result fields and loader arguments without aliases."""
    assert [item.name for item in fields(LocalExplanationData)] == [
        "model_id",
        "prediction_id",
        "instance",
        "feature_names",
        "background",
        "background_output",
        "output_names",
        "input_tensor_name",
        "output_tensor_name",
    ]
    assert not hasattr(LocalExplanationData, "targets")
    parameters = signature(load_local_explanation_data).parameters
    assert list(parameters) == [
        "model_id",
        "prediction_id",
        "max_background_rows",
        "include_stored_output",
        "storage_interface",
    ]
    assert (
        parameters["max_background_rows"].default
        is parameters["max_background_rows"].empty
    )
    assert (
        parameters["include_stored_output"].default
        is parameters["include_stored_output"].empty
    )


@pytest.mark.asyncio
async def test_loader_finds_target_outside_newest_background_window() -> None:
    """Find an old target while selecting only the newest organic background."""
    ids = ["target", "old-organic", "synthetic", "organic-3", "organic-4"]
    storage = _storage(
        ids,
        tags=[[], [], [SYNTHETIC_TAG], [], []],
    )

    data = await load_local_explanation_data(
        "model",
        "target",
        2,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.instance.tolist() == [0.0, 100.0]
    assert data.background[:, 0].tolist() == [3.0, 4.0]


@pytest.mark.asyncio
async def test_loader_excludes_target_and_interleaved_synthetic_rows() -> None:
    """Filter target and both current and legacy synthetic tags in storage order."""
    ids = [
        "organic-0",
        "target",
        "synthetic-2",
        "organic-3",
        "synthetic-4",
        "organic-5",
    ]
    tags = [[], [], ["synthetic"], [], [b"synthetic"], []]
    storage = _storage(ids, tags=tags)

    data = await load_local_explanation_data(
        "model",
        "target",
        10,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.background[:, 0].tolist() == [0.0, 3.0, 5.0]


@pytest.mark.asyncio
async def test_loader_decodes_byte_valued_ids() -> None:
    """Compare requested text IDs with byte-valued metadata IDs."""
    storage = _storage([b"target", b"organic"])

    data = await load_local_explanation_data(
        "model",
        "target",
        10,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.prediction_id == "target"
    assert data.instance.tolist() == [0.0, 100.0]


@pytest.mark.asyncio
async def test_loader_decodes_mariadb_shaped_zero_dimensional_byte_id() -> None:
    """Unwrap MariaDB's zero-dimensional NumPy scalar before ID comparison."""
    storage = _storage(["organic"])
    storage.metadata = np.empty((2, 4), dtype=object)
    storage.metadata[0] = [
        np.asarray(b"target"),
        "2026-09-22T00:00:00",
        0.0,
        [],
    ]
    storage.metadata[1] = [
        b"organic",
        "2026-09-22T00:00:00",
        0.0,
        [],
    ]
    storage.inputs = np.array([[1.0, 2.0], [3.0, 4.0]])

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.prediction_id == "target"
    assert data.instance.tolist() == [1.0, 2.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_id", "requested_id"),
    [
        (None, "None"),
        (np.nan, "nan"),
        (np.inf, "inf"),
        ("", ""),
        ("   ", "   "),
    ],
)
async def test_loader_rejects_missing_blank_or_nonfinite_metadata_ids(
    stored_id: object,
    requested_id: str,
) -> None:
    """Do not let stringified missing metadata IDs select a stored row."""
    storage = _storage([stored_id, "organic"])

    with pytest.raises(ValueError, match="prediction ID"):
        await load_local_explanation_data(
            "model",
            requested_id,
            1,
            include_stored_output=False,
            storage_interface=storage,
        )


@pytest.mark.asyncio
async def test_loader_accepts_zero_dimensional_bytes_and_zero_value_ids() -> None:
    """Preserve valid zero-dimensional byte and numeric metadata IDs."""
    byte_storage = _storage([np.asarray(np.bytes_("byte-id")), "organic"])
    byte_data = await load_local_explanation_data(
        "model",
        "byte-id",
        1,
        include_stored_output=False,
        storage_interface=byte_storage,
    )
    assert byte_data.prediction_id == "byte-id"

    zero_storage = _storage([0, "organic"])
    zero_data = await load_local_explanation_data(
        "model",
        "0",
        1,
        include_stored_output=False,
        storage_interface=zero_storage,
    )
    assert zero_data.prediction_id == "0"


@pytest.mark.asyncio
async def test_loader_continues_metadata_scan_after_target_match() -> None:
    """Detect duplicate IDs in a later bounded metadata chunk."""
    row_count = 1001
    ids = ["target", *[f"row-{index}" for index in range(1, row_count - 1)], "target"]
    storage = _storage(ids)

    with pytest.raises(ValueError, match="ambiguous"):
        await load_local_explanation_data(
            "model",
            "target",
            1,
            include_stored_output=False,
            storage_interface=storage,
        )

    metadata_reads = [
        read for read in storage.reads if read[0].endswith(METADATA_SUFFIX)
    ]
    assert len(metadata_reads) == 2
    assert all(read[2] is not None and read[2] <= 1000 for read in metadata_reads)


@pytest.mark.asyncio
async def test_loader_scans_past_the_former_million_row_boundary() -> None:
    """Find a target in a later chunk instead of silently truncating the scan."""
    storage = _VirtualStorage(target_indices={1_000_000})

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.instance.tolist() == [1_000_000.0, 1_000_100.0]
    assert data.background[:, 0].tolist() == [999_999.0]


@pytest.mark.asyncio
async def test_loader_scans_to_older_organic_background_rows() -> None:
    """Find organic background after newer chunks are all synthetic."""
    storage = _VirtualStorage(
        target_indices={1_000_000},
        organic_indices={0},
    )

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.background.tolist() == [[0.0, 100.0]]


@pytest.mark.asyncio
async def test_loader_detects_duplicate_id_after_the_former_boundary() -> None:
    """Continue duplicate detection through every bounded metadata chunk."""
    storage = _VirtualStorage(target_indices={0, 1_000_000})

    with pytest.raises(ValueError, match="ambiguous"):
        await load_local_explanation_data(
            "model",
            "target",
            1,
            include_stored_output=False,
            storage_interface=storage,
        )


@pytest.mark.asyncio
async def test_loader_rejects_missing_prediction_id() -> None:
    """Reject a target that is not present instead of selecting a fallback row."""
    storage = _storage(["other"])

    with pytest.raises(ValueError, match="not found"):
        await load_local_explanation_data(
            "model",
            "target",
            1,
            include_stored_output=False,
            storage_interface=storage,
        )


@pytest.mark.asyncio
async def test_loader_rejects_duplicate_input_aliases() -> None:
    """Reject ambiguous feature-to-column mappings."""
    storage = _storage(["target", "organic"], input_names=["feature", "feature"])

    with pytest.raises(ValueError, match="aliases"):
        await load_local_explanation_data(
            "model",
            "target",
            1,
            include_stored_output=False,
            storage_interface=storage,
        )


@pytest.mark.asyncio
async def test_loader_preserves_feature_order_and_row_alignment() -> None:
    """Keep aliased input order and pair the target with its metadata row."""
    storage = _storage(
        ["first", "target", "last"],
        inputs=np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]),
        input_names=["second-column", "first-column"],
    )

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.feature_names == ["second-column", "first-column"]
    assert data.instance.tolist() == [20.0, 21.0]
    assert data.background.tolist() == [[30.0, 31.0]]


@pytest.mark.asyncio
async def test_loader_enforces_bounded_background_size_and_storage_order() -> None:
    """Return at most the requested newest organic rows in original order."""
    ids = [f"row-{index}" for index in range(8)]
    tags = [[], ["synthetic"], [], [], [SYNTHETIC_TAG], [], [], []]
    storage = _storage(ids, tags=tags)

    data = await load_local_explanation_data(
        "model",
        "row-0",
        3,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.background.shape == (3, 2)
    assert data.background[:, 0].tolist() == [5.0, 6.0, 7.0]


@pytest.mark.asyncio
async def test_model_mode_does_not_read_or_require_stored_outputs() -> None:
    """Allow MODEL data loading when the output dataset is absent."""
    storage = _storage(
        ["target", "organic"],
        fail_get_metadata=True,
    )

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.background_output is None
    assert data.input_tensor_name == "input"
    assert data.output_tensor_name == "output"
    assert storage.metadata_accesses == 0
    assert not any(read[0].endswith(OUTPUT_SUFFIX) for read in storage.reads)


@pytest.mark.asyncio
async def test_loader_filters_invalid_excluded_input_rows_before_validation() -> None:
    """Do not reject invalid feature values on target or synthetic rows."""
    storage = _storage(
        ["target", "synthetic", "organic"],
        tags=[[], [SYNTHETIC_TAG], []],
        inputs=np.array(
            [[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]],
            dtype=float,
        ),
    )

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=False,
        storage_interface=storage,
    )

    assert data.background.tolist() == [[3.0, 4.0]]


@pytest.mark.asyncio
async def test_loader_filters_invalid_excluded_output_rows_before_validation() -> None:
    """Do not reject invalid labels on target or synthetic rows in SURROGATE mode."""
    storage = _storage(
        ["target", "synthetic", "organic"],
        tags=[[], [SYNTHETIC_TAG], []],
        outputs=np.array([[0.0], [np.nan], [1.0]], dtype=float),
        output_names=["label"],
    )

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=True,
        storage_interface=storage,
    )

    assert data.background_output is not None
    assert data.background_output.tolist() == [[1.0]]


@pytest.mark.asyncio
async def test_surrogate_mode_requires_aligned_stored_outputs() -> None:
    """Load aligned output labels only for explicit surrogate mode."""
    storage = _storage(
        ["target", "organic", "synthetic"],
        tags=[[], [], [SYNTHETIC_TAG]],
        outputs=np.array([[9.0], [8.0], [7.0]]),
        output_names=["label"],
    )

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=True,
        storage_interface=storage,
    )

    assert data.output_names == ["label"]
    assert data.background_output is not None
    assert data.background_output.tolist() == [[8.0]]


@pytest.mark.asyncio
async def test_surrogate_mode_reports_missing_stored_outputs_with_typed_error() -> None:
    """Identify an absent output dataset without using a broad error message."""
    storage = _storage(["target", "organic"])

    with pytest.raises(
        LocalExplanationMissingOutputError,
        match="Stored output labels are unavailable",
    ):
        await load_local_explanation_data(
            "model",
            "target",
            1,
            include_stored_output=True,
            storage_interface=storage,
        )


@pytest.mark.asyncio
async def test_loader_rejects_input_metadata_row_misalignment() -> None:
    """Do not index input rows using an unaligned metadata row number."""
    storage = _storage(["target", "organic"], input_rows=3)

    with pytest.raises(ValueError, match="input and metadata"):
        await load_local_explanation_data(
            "model",
            "target",
            1,
            include_stored_output=False,
            storage_interface=storage,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inputs",
    [
        np.array([[1.0, np.nan], [3.0, 4.0]], dtype=float),
        np.array([[1.0, np.inf], [3.0, 4.0]], dtype=float),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=object),
        np.array([["1", "2"], ["3", "4"]]),
        np.ones((2, 1, 2), dtype=float),
    ],
)
async def test_loader_rejects_invalid_model_feature_arrays(inputs: np.ndarray) -> None:
    """Reject non-finite, non-numeric, and higher-rank model features."""
    storage = _storage(["target", "organic"], inputs=inputs)

    with pytest.raises(ValueError, match="input"):
        await load_local_explanation_data(
            "model",
            "target",
            1,
            include_stored_output=False,
            storage_interface=storage,
        )


@pytest.mark.asyncio
async def test_loader_normalizes_collapsed_single_row_reads() -> None:
    """Restore one-row backend reads without flattening the model feature width."""
    storage = _storage(
        ["target", "organic"],
        inputs=np.array([[1.0, 2.0], [3.0, 4.0]]),
        outputs=np.array([[0.0], [1.0]]),
        collapse_single_row=True,
    )

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=True,
        storage_interface=storage,
    )

    assert data.instance.shape == (2,)
    assert data.background.shape == (1, 2)
    assert data.background_output is not None
    assert data.background_output.shape == (1, 1)


@pytest.mark.asyncio
async def test_loader_returns_tensor_name_hints_when_storage_provides_them() -> None:
    """Preserve optional persisted tensor names as orchestration hints."""
    storage = _storage(
        ["target", "organic"],
        outputs=np.array([[0.0], [1.0]]),
        tensor_names={"inputTensorName": "features", "outputTensorName": "score"},
    )

    data = await load_local_explanation_data(
        "model",
        "target",
        1,
        include_stored_output=True,
        storage_interface=storage,
    )

    assert data.input_tensor_name == "features"
    assert data.output_tensor_name == "score"
