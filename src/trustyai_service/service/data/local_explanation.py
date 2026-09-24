"""Bounded, schema-aware storage loading for local explanations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from trustyai_service.service.constants import (
    INPUT_SUFFIX,
    METADATA_ID_COL,
    METADATA_SUFFIX,
    METADATA_TAGS_COL,
    METADATA_TIMESTAMP_COL,
    OUTPUT_SUFFIX,
    SYNTHETIC_TAG,
)
from trustyai_service.service.data.storage import get_global_storage_interface

if TYPE_CHECKING:
    from trustyai_service.service.data.storage.storage_interface import (
        StorageInterface,
    )

_CHUNK_SIZE = 1000
_MATRIX_RANK = 2
_LEGACY_SYNTHETIC_TAG = "synthetic"
_SYNTHETIC_TAGS = frozenset({SYNTHETIC_TAG, _LEGACY_SYNTHETIC_TAG})


class LocalExplanationMissingOutputError(ValueError):
    """Indicate that the stored output dataset is absent for surrogate mode."""


@dataclass(frozen=True)
class LocalExplanationData:
    """Stored target instance and bounded organic background data."""

    model_id: str
    prediction_id: str
    instance: np.ndarray
    feature_names: list[str]
    background: np.ndarray
    background_output: np.ndarray | None
    output_names: list[str]
    input_tensor_name: str | None
    output_tensor_name: str | None


@dataclass(frozen=True)
class _BackgroundSpec:
    """Dataset and shape information for one bounded background scan."""

    input_dataset: str
    metadata_dataset: str
    output_dataset: str | None
    target_index: int
    total_rows: int
    input_width: int
    output_width: int
    max_background_rows: int


def _decode_id_value(value: object) -> str:
    """Decode a stored ID, including zero-dimensional NumPy scalar arrays."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            msg = "Stored text metadata is not valid UTF-8"
            raise ValueError(msg) from exc
    return str(value)


def _decode_metadata_id(value: object) -> str:
    """Decode and validate one stored metadata prediction ID."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if value is None:
        msg = "Stored metadata contains an invalid prediction ID"
        raise ValueError(msg)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        msg = "Stored metadata contains an invalid prediction ID"
        raise ValueError(msg)

    decoded = _decode_id_value(value)
    if not decoded.strip():
        msg = "Stored metadata contains an invalid prediction ID"
        raise ValueError(msg)
    return decoded


def _decode_text(value: object) -> str:
    """Decode stored byte text while preserving ordinary string values."""
    return _decode_id_value(value)


def _validate_metadata_row(row: np.ndarray) -> None:
    """Require the shared metadata columns before interpreting a row."""
    if row.shape[0] <= METADATA_TIMESTAMP_COL:
        msg = "Stored metadata is missing required columns"
        raise ValueError(msg)


def _iter_tag_values(value: object) -> list[object]:
    """Flatten the list-like values used by storage for one tag cell."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values: list[object] = []
        for item in value:
            values.extend(_iter_tag_values(item))
        return values
    return [] if value is None else [value]


def _is_synthetic(value: object) -> bool:
    """Return whether a metadata tag cell carries a synthetic-row tag."""
    return any(_decode_text(tag) in _SYNTHETIC_TAGS for tag in _iter_tag_values(value))


def _normalize_matrix(
    value: object,
    *,
    expected_rows: int,
    expected_columns: int | None,
    name: str,
) -> np.ndarray:
    """Normalize backend row reads without flattening multi-row data."""
    if value is None:
        msg = f"Stored {name} data is unavailable"
        raise ValueError(msg)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        msg = f"Stored {name} data is not a valid matrix"
        raise ValueError(msg) from exc

    if array.ndim == 1:
        if expected_rows == 1:
            array = array.reshape(1, -1)
        elif len(array) == expected_rows:
            array = array.reshape(expected_rows, 1)
        else:
            msg = f"Stored {name} data must preserve its multi-row shape"
            raise ValueError(msg)
    if array.ndim != _MATRIX_RANK:
        msg = f"Stored {name} data must be a two-dimensional matrix"
        raise ValueError(msg)
    if array.shape[0] != expected_rows:
        msg = f"Stored {name} rows are misaligned"
        raise ValueError(msg)
    if expected_columns is not None and array.shape[1] != expected_columns:
        msg = f"Stored {name} columns do not match their aliases"
        raise ValueError(msg)
    if array.shape[0] == 0:
        msg = f"Stored {name} data must be non-empty"
        raise ValueError(msg)
    return array


def _numeric_matrix(
    value: object,
    *,
    expected_rows: int,
    expected_columns: int | None,
    name: str,
) -> np.ndarray:
    """Validate and return finite numeric storage rows as float64."""
    array = _normalize_matrix(
        value,
        expected_rows=expected_rows,
        expected_columns=expected_columns,
        name=name,
    )
    if array.dtype.kind not in "biuf":
        msg = f"Stored {name} data must be numeric"
        raise ValueError(msg)
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = f"Stored {name} data must be numeric"
        raise ValueError(msg) from exc
    if not np.isfinite(numeric).all():
        msg = f"Stored {name} data must be finite"
        raise ValueError(msg)
    return numeric


def _row_count(value: object, *, name: str) -> int:
    """Validate a storage row count and return a Python integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f"Stored {name} row count is invalid"
        raise TypeError(msg)
    count = int(value)
    if count < 0:
        msg = f"Stored {name} row count is invalid"
        raise ValueError(msg)
    return count


async def _metadata_target_index(
    storage: StorageInterface,
    dataset_name: str,
    prediction_id: str,
    metadata_rows: int,
) -> int:
    """Scan bounded metadata chunks and reject missing or duplicate IDs."""
    matches: list[int] = []
    scan_rows = metadata_rows
    for offset in range(0, scan_rows, _CHUNK_SIZE):
        requested_rows = min(_CHUNK_SIZE, scan_rows - offset)
        raw_metadata = await storage.read_data(
            dataset_name,
            start_row=offset,
            n_rows=requested_rows,
        )
        metadata = _normalize_matrix(
            raw_metadata,
            expected_rows=requested_rows,
            expected_columns=None,
            name="metadata",
        )
        for row_offset, row in enumerate(metadata):
            _validate_metadata_row(row)
            if row.shape[0] <= METADATA_ID_COL:
                msg = "Stored metadata is missing prediction IDs"
                raise ValueError(msg)
            if _decode_metadata_id(row[METADATA_ID_COL]) == prediction_id:
                matches.append(offset + row_offset)

    if len(matches) > 1:
        msg = "Prediction ID is ambiguous"
        raise ValueError(msg)
    if not matches:
        msg = "Prediction ID was not found"
        raise ValueError(msg)
    return matches[0]


def _background_offsets(total_rows: int) -> range:
    """Return newest-first chunk offsets for a chunked background scan."""
    if total_rows == 0:
        return range(0)
    first_chunk = ((total_rows - 1) // _CHUNK_SIZE) * _CHUNK_SIZE
    return range(first_chunk, -1, -_CHUNK_SIZE)


async def _read_background_chunk(
    storage: StorageInterface,
    spec: _BackgroundSpec,
    offset: int,
    expected_rows: int,
    selected_rows: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
    """Read and filter one newest-first storage chunk."""
    raw_metadata = await storage.read_data(
        spec.metadata_dataset,
        start_row=offset,
        n_rows=expected_rows,
    )
    metadata = _normalize_matrix(
        raw_metadata,
        expected_rows=expected_rows,
        expected_columns=None,
        name="metadata",
    )
    row_count = metadata.shape[0]

    remaining = spec.max_background_rows - selected_rows
    selected_indices: list[int] = []
    for row_offset in range(row_count - 1, -1, -1):
        row = metadata[row_offset]
        _validate_metadata_row(row)
        absolute_index = offset + row_offset
        tags = row[METADATA_TAGS_COL] if row.shape[0] > METADATA_TAGS_COL else []
        if absolute_index == spec.target_index or _is_synthetic(tags):
            continue
        selected_indices.append(row_offset)
        if len(selected_indices) >= remaining:
            break
    selected_indices.sort()
    if not selected_indices:
        return None, None, []

    raw_input = await storage.read_data(
        spec.input_dataset,
        start_row=offset,
        n_rows=expected_rows,
    )
    input_matrix = _normalize_matrix(
        raw_input,
        expected_rows=expected_rows,
        expected_columns=spec.input_width,
        name="input",
    )
    selected_input = _numeric_matrix(
        input_matrix[selected_indices],
        expected_rows=len(selected_indices),
        expected_columns=spec.input_width,
        name="input",
    )

    selected_output: np.ndarray | None = None
    if spec.output_dataset is not None:
        raw_output = await storage.read_data(
            spec.output_dataset,
            start_row=offset,
            n_rows=expected_rows,
        )
        output_matrix = _normalize_matrix(
            raw_output,
            expected_rows=expected_rows,
            expected_columns=spec.output_width,
            name="output",
        )
        selected_output = _numeric_matrix(
            output_matrix[selected_indices],
            expected_rows=len(selected_indices),
            expected_columns=spec.output_width,
            name="output",
        )
    return selected_input, selected_output, selected_indices


async def _load_background(
    storage: StorageInterface,
    spec: _BackgroundSpec,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read newest organic rows and restore them to storage order."""
    input_chunks: list[np.ndarray] = []
    output_chunks: list[np.ndarray] = []
    selected_rows = 0

    for offset in _background_offsets(spec.total_rows):
        expected_rows = min(_CHUNK_SIZE, spec.total_rows - offset)
        input_matrix, output_matrix, selected_indices = await _read_background_chunk(
            storage,
            spec,
            offset,
            expected_rows,
            selected_rows,
        )
        if not selected_indices:
            continue
        if input_matrix is None:
            msg = "Stored input data is unavailable"
            raise ValueError(msg)
        input_chunks.append(input_matrix)
        if output_matrix is not None:
            output_chunks.append(output_matrix)
        selected_rows += len(selected_indices)

        if selected_rows >= spec.max_background_rows:
            break

    if not input_chunks:
        msg = "No organic background data is available"
        raise ValueError(msg)

    input_chunks.reverse()
    background = np.concatenate(input_chunks, axis=0)
    if spec.output_dataset is None:
        return background, None

    output_chunks.reverse()
    return background, np.concatenate(output_chunks, axis=0)


async def _tensor_name_hints(
    storage: StorageInterface,
    model_id: str,
) -> tuple[str | None, str | None]:
    """Read optional persisted tensor-name hints without requiring them."""
    getter = getattr(storage, "get_metadata", None)
    if not callable(getter):
        return "input", "output"
    try:
        metadata_getter = cast("Callable[[str], Awaitable[object]]", getter)
        metadata = await metadata_getter(model_id)
    except (AttributeError, KeyError, TypeError, ValueError):
        return "input", "output"
    if metadata is None:
        return "input", "output"

    if isinstance(metadata, Mapping):
        input_name = metadata.get("inputTensorName") or metadata.get(
            "input_tensor_name"
        )
        output_name = metadata.get("outputTensorName") or metadata.get(
            "output_tensor_name"
        )
    else:
        input_name = getattr(metadata, "input_tensor_name", None)
        output_name = getattr(metadata, "output_tensor_name", None)
        if input_name is None:
            input_getter = getattr(metadata, "get_input_tensor_name", None)
            input_name = input_getter() if callable(input_getter) else None
        if output_name is None:
            output_getter = getattr(metadata, "get_output_tensor_name", None)
            output_name = output_getter() if callable(output_getter) else None

    return (
        _decode_text(input_name) if input_name else "input",
        _decode_text(output_name) if output_name else "output",
    )


async def _load_target_context(
    storage: StorageInterface,
    *,
    input_dataset: str,
    metadata_dataset: str,
    prediction_id: str,
) -> tuple[int, int, list[str], np.ndarray]:
    """Load aligned target metadata and input rows."""
    if not await storage.dataset_exists(
        metadata_dataset
    ) or not await storage.dataset_exists(input_dataset):
        msg = "No stored data exists for the requested model"
        raise LookupError(msg)

    metadata_rows = _row_count(
        await storage.dataset_rows(metadata_dataset), name="metadata"
    )
    input_rows = _row_count(await storage.dataset_rows(input_dataset), name="input")
    if metadata_rows != input_rows:
        msg = "Stored input and metadata rows are misaligned"
        raise ValueError(msg)
    target_index = await _metadata_target_index(
        storage,
        metadata_dataset,
        prediction_id,
        metadata_rows,
    )
    feature_names = [
        _decode_text(name)
        for name in await storage.get_aliased_column_names(input_dataset)
    ]
    if not feature_names or len(set(feature_names)) != len(feature_names):
        msg = "Stored input aliases are missing or ambiguous"
        raise ValueError(msg)
    raw_instance = await storage.read_data(
        input_dataset,
        start_row=target_index,
        n_rows=1,
    )
    instance = _numeric_matrix(
        raw_instance,
        expected_rows=1,
        expected_columns=len(feature_names),
        name="input",
    )
    return input_rows, target_index, feature_names, instance


async def _load_output_context(
    storage: StorageInterface,
    *,
    output_dataset: str | None,
    input_rows: int,
) -> tuple[list[str], int]:
    """Validate aligned stored output labels when surrogate mode needs them."""
    if output_dataset is None:
        return [], 0
    if not await storage.dataset_exists(output_dataset):
        msg = "Stored output labels are unavailable"
        raise LocalExplanationMissingOutputError(msg)
    output_rows = _row_count(await storage.dataset_rows(output_dataset), name="output")
    if output_rows != input_rows:
        msg = "Stored input and output rows are misaligned"
        raise ValueError(msg)
    output_names = [
        _decode_text(name)
        for name in await storage.get_aliased_column_names(output_dataset)
    ]
    if not output_names or len(set(output_names)) != len(output_names):
        msg = "Stored output aliases are missing or ambiguous"
        raise ValueError(msg)
    return output_names, len(output_names)


async def load_local_explanation_data(
    model_id: str,
    prediction_id: str,
    max_background_rows: int,
    *,
    include_stored_output: bool,
    storage_interface: StorageInterface | None = None,
) -> LocalExplanationData:
    """Load one target row and a bounded organic background from storage."""
    if isinstance(max_background_rows, bool) or not isinstance(
        max_background_rows, int
    ):
        msg = "max_background_rows must be a positive integer"
        raise TypeError(msg)
    if max_background_rows < 1:
        msg = "max_background_rows must be positive"
        raise ValueError(msg)
    if not isinstance(include_stored_output, bool):
        msg = "include_stored_output must be a boolean"
        raise TypeError(msg)

    storage = (
        get_global_storage_interface()
        if storage_interface is None
        else storage_interface
    )
    input_dataset = model_id + INPUT_SUFFIX
    metadata_dataset = model_id + METADATA_SUFFIX

    normalized_prediction_id = _decode_id_value(prediction_id)
    (
        input_rows,
        target_index,
        feature_names,
        instance_matrix,
    ) = await _load_target_context(
        storage,
        input_dataset=input_dataset,
        metadata_dataset=metadata_dataset,
        prediction_id=normalized_prediction_id,
    )

    output_dataset = model_id + OUTPUT_SUFFIX if include_stored_output else None
    output_names, output_width = await _load_output_context(
        storage,
        output_dataset=output_dataset,
        input_rows=input_rows,
    )

    background, background_output = await _load_background(
        storage,
        _BackgroundSpec(
            input_dataset=input_dataset,
            metadata_dataset=metadata_dataset,
            output_dataset=output_dataset,
            target_index=target_index,
            total_rows=input_rows,
            input_width=len(feature_names),
            output_width=output_width,
            max_background_rows=max_background_rows,
        ),
    )
    if include_stored_output:
        input_tensor_name, output_tensor_name = await _tensor_name_hints(
            storage, model_id
        )
    else:
        input_tensor_name, output_tensor_name = "input", "output"
    return LocalExplanationData(
        model_id=model_id,
        prediction_id=normalized_prediction_id,
        instance=instance_matrix[0],
        feature_names=feature_names,
        background=background,
        background_output=background_output,
        output_names=output_names,
        input_tensor_name=input_tensor_name,
        output_tensor_name=output_tensor_name,
    )
