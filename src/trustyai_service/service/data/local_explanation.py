"""Bounded, schema-aware storage loading for local explanations."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from trustyai_service.service.constants import (
    INPUT_SUFFIX,
    METADATA_ID_COL,
    METADATA_SUFFIX,
    METADATA_TAGS_COL,
    OUTPUT_SUFFIX,
    SYNTHETIC_TAG,
)
from trustyai_service.service.data.exceptions import StorageReadError
from trustyai_service.service.data.storage import get_global_storage_interface
from trustyai_service.service.data.storage.exceptions import StorageError
from trustyai_service.service.explainers.local.model_provider import (
    LocalDataError,
    LocalDataNotFoundError,
)

_CHUNK_SIZE = 1000
_MAX_SCAN_ROWS = 1_000_000
_DEFAULT_TRAINING_ROWS = 10_000
_MATRIX_RANK = 2


@dataclass(frozen=True)
class LocalExplanationData:
    """Target instance and bounded organic background for one explanation."""

    model_id: str
    prediction_id: str
    instance: np.ndarray
    feature_names: list[str]
    background: np.ndarray
    background_output: np.ndarray | None
    output_names: list[str] = field(default_factory=list)
    input_tensor_name: str | None = None
    output_tensor_name: str | None = None

    @property
    def targets(self) -> np.ndarray | None:
        """Compatibility alias for older service callers."""
        return self.background_output


@dataclass(frozen=True)
class _BackgroundSpec:
    """Dataset identifiers and row limits for one background scan."""

    input_dataset: str
    metadata_dataset: str
    output_dataset: str | None
    target_index: int
    training_rows: int
    input_width: int
    output_names: list[str]


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def _rows(value: object, *, name: str) -> np.ndarray:
    if value is None:
        msg = f"Stored {name} data is unavailable"
        raise LocalDataError(msg)
    array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != _MATRIX_RANK or not len(array):
        msg = f"Stored {name} data must be a non-empty matrix"
        raise LocalDataError(msg)
    return array


def _numeric_rows(value: object, *, name: str) -> np.ndarray:
    array = _rows(value, name=name)
    # Storage values are already typed by the storage backend.  Reject object
    # and string arrays instead of accepting silently coercible values: doing
    # so prevents malformed payloads from becoming valid model features.
    if array.dtype.kind not in "biuf":
        msg = f"Stored {name} data must be numeric"
        raise LocalDataError(msg)
    try:
        result = array.astype(float)
    except (TypeError, ValueError) as exc:
        msg = f"Stored {name} data must be numeric"
        raise LocalDataError(msg) from exc
    if not np.isfinite(result).all():
        msg = f"Stored {name} data must be finite"
        raise LocalDataError(msg)
    return result


def _chunk_matrix(
    value: object,
    row_count: int | None,
    *,
    name: str,
    column_count: int | None = None,
) -> np.ndarray:
    if value is None:
        msg = f"Stored {name} data is unavailable"
        raise LocalDataError(msg)
    array = np.asarray(value)
    if array.ndim == 1:
        if row_count is None:
            if column_count is None or len(array) % column_count != 0:
                msg = f"Stored {name} rows are misaligned"
                raise LocalDataError(msg)
            row_count = len(array) // column_count
        if row_count == 1:
            array = array.reshape(1, -1)
        elif len(array) == row_count:
            array = array.reshape(row_count, 1)
        elif len(array) % row_count == 0:
            array = array.reshape(row_count, -1)
        else:
            msg = f"Stored {name} rows are misaligned"
            raise LocalDataError(msg)
    if array.ndim != _MATRIX_RANK or (
        row_count is not None and len(array) != row_count
    ):
        msg = f"Stored {name} rows are misaligned"
        raise LocalDataError(msg)
    return array


def _synthetic(value: object) -> bool:
    if hasattr(value, "tolist"):
        value = value.tolist()
    values = value if isinstance(value, (list, tuple)) else [value]
    return any(_text(item) in {SYNTHETIC_TAG, "synthetic"} for item in values)


async def _metadata_matches(
    storage: object, dataset: str, prediction_id: str, row_limit: int
) -> int:
    matches: list[int] = []
    for offset in range(0, min(row_limit, _MAX_SCAN_ROWS), _CHUNK_SIZE):
        rows = await storage.read_data(dataset, start_row=offset, n_rows=_CHUNK_SIZE)
        if rows is None or len(rows) == 0:
            break
        matrix = np.asarray(rows)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        for index, row in enumerate(matrix):
            if len(row) <= METADATA_ID_COL:
                msg = "Stored metadata is missing prediction IDs"
                raise LocalDataError(msg)
            if _text(row[METADATA_ID_COL]) == prediction_id:
                matches.append(offset + index)
    if len(matches) > 1:
        msg = "Prediction ID is ambiguous"
        raise LocalDataError(msg)
    if not matches:
        msg = "Prediction ID was not found"
        raise LocalDataError(msg)
    return matches[0]


async def _load_background(
    storage: object,
    spec: _BackgroundSpec,
) -> tuple[np.ndarray, np.ndarray | None]:
    input_chunks: list[np.ndarray] = []
    output_chunks: list[np.ndarray] = []
    selected = 0
    offsets, has_count, total_rows = await _background_offsets(
        storage, spec.input_dataset
    )
    for offset in offsets:
        expected_rows = (
            min(_CHUNK_SIZE, total_rows - offset) if total_rows is not None else None
        )
        chunk = await _read_background_chunk(
            storage, spec, offset, selected, expected_rows=expected_rows
        )
        if chunk is None:
            continue
        input_matrix, output_matrix, selected_indices = chunk
        selected += len(selected_indices)
        if selected_indices:
            input_chunks.append(input_matrix[selected_indices])
            if output_matrix is not None:
                output_chunks.append(output_matrix[selected_indices])
        if selected >= spec.training_rows:
            break
    if not input_chunks:
        msg = "No organic background data is available"
        raise LocalDataError(msg)
    ordered_inputs = input_chunks[::-1] if has_count else input_chunks
    ordered_outputs = output_chunks[::-1] if has_count else output_chunks
    background = np.concatenate(ordered_inputs, axis=0)
    if spec.output_dataset is None:
        return background, None
    return background, np.concatenate(ordered_outputs, axis=0)


async def _background_offsets(
    storage: object, input_dataset: str
) -> tuple[range, bool, int | None]:
    """Return newest-first chunk offsets when the backend exposes row counts."""
    has_count = True
    try:
        total_rows = await storage.dataset_rows(input_dataset)
    except (AttributeError, TypeError):
        has_count = False
        total_rows = _MAX_SCAN_ROWS
    if not isinstance(total_rows, int) or total_rows < 0:
        has_count = False
        total_rows = _MAX_SCAN_ROWS
    total_rows = min(total_rows, _MAX_SCAN_ROWS)
    if has_count and total_rows:
        first = ((total_rows - 1) // _CHUNK_SIZE) * _CHUNK_SIZE
        return range(first, -1, -_CHUNK_SIZE), True, total_rows
    if has_count:
        return range(0), False, 0
    return range(0, _MAX_SCAN_ROWS, _CHUNK_SIZE), False, None


async def _read_background_chunk(
    storage: object,
    spec: _BackgroundSpec,
    offset: int,
    selected: int,
    *,
    expected_rows: int | None,
) -> tuple[np.ndarray, np.ndarray | None, list[int]] | None:
    """Read, validate, and filter one storage chunk."""
    input_chunk = await storage.read_data(
        spec.input_dataset, start_row=offset, n_rows=_CHUNK_SIZE
    )
    if input_chunk is None or len(input_chunk) == 0:
        return None
    metadata_chunk = await storage.read_data(
        spec.metadata_dataset, start_row=offset, n_rows=_CHUNK_SIZE
    )
    input_matrix = _numeric_rows(
        _chunk_matrix(
            input_chunk,
            expected_rows,
            name="input",
            column_count=spec.input_width,
        ),
        name="input",
    )
    row_count = len(input_matrix)
    metadata_matrix = _chunk_matrix(metadata_chunk, row_count, name="metadata")
    output_matrix = await _read_output_chunk(storage, spec, offset, row_count)
    remaining = max(0, spec.training_rows - selected)
    selected_indices: list[int] = []
    for index, metadata in enumerate(metadata_matrix):
        absolute = offset + index
        if absolute == spec.target_index or _synthetic(
            metadata[METADATA_TAGS_COL] if len(metadata) > METADATA_TAGS_COL else []
        ):
            continue
        selected_indices.append(index)
        if len(selected_indices) >= remaining:
            break
    return input_matrix, output_matrix, selected_indices


async def _read_output_chunk(
    storage: object, spec: _BackgroundSpec, offset: int, row_count: int
) -> np.ndarray | None:
    """Read and validate aligned organic labels when surrogate mode needs them."""
    if spec.output_dataset is None:
        return None
    output_chunk = await storage.read_data(
        spec.output_dataset, start_row=offset, n_rows=_CHUNK_SIZE
    )
    output_matrix = _numeric_rows(
        _chunk_matrix(output_chunk, row_count, name="output"), name="output"
    )
    if output_matrix.shape[1] != len(spec.output_names):
        msg = "Stored output columns do not match their aliases"
        raise LocalDataError(msg)
    return output_matrix


async def load_local_explanation_data(
    model: str,
    prediction_id: str,
    *args: object,
    **options: object,
) -> LocalExplanationData:
    """Load and validate one target row plus bounded background data."""
    if len(args) > 1:
        msg = "Expected at most one positional argument after prediction_id"
        raise TypeError(msg)
    if args and "max_background_rows" in options:
        msg = "Multiple values for argument: max_background_rows"
        raise TypeError(msg)
    max_background_rows = cast(
        "int | None",
        args[0] if args else options.pop("max_background_rows", None),
    )
    include_stored_output = cast(
        "bool | None", options.pop("include_stored_output", None)
    )
    include_targets = cast("bool | None", options.pop("include_targets", None))
    n_training_rows = cast("int | None", options.pop("n_training_rows", None))
    storage_interface = cast("object | None", options.pop("storage_interface", None))
    if options:
        names = ", ".join(sorted(options))
        msg = f"Unexpected keyword argument(s): {names}"
        raise TypeError(msg)
    max_background_rows, include_stored_output = _resolve_loader_options(
        max_background_rows,
        include_stored_output=include_stored_output,
        include_targets=include_targets,
        n_training_rows=n_training_rows,
    )
    storage = storage_interface or get_global_storage_interface()
    metadata_name = model + METADATA_SUFFIX
    input_name = model + INPUT_SUFFIX
    input_rows, row_index, feature_names, instance = await _load_target_context(
        storage, metadata_name, input_name, str(prediction_id)
    )
    output_dataset = model + OUTPUT_SUFFIX if include_stored_output else None
    output_names = await _load_output_context(storage, output_dataset, input_rows)
    input_tensor_name, output_tensor_name = await _tensor_name_hints(storage, model)
    background, targets = await _load_background(
        storage,
        _BackgroundSpec(
            input_dataset=input_name,
            metadata_dataset=metadata_name,
            output_dataset=output_dataset,
            target_index=row_index,
            training_rows=max_background_rows,
            input_width=len(feature_names),
            output_names=output_names,
        ),
    )
    return LocalExplanationData(
        model_id=model,
        prediction_id=str(prediction_id),
        instance=instance[0],
        feature_names=feature_names,
        background=background,
        background_output=targets,
        output_names=output_names,
        input_tensor_name=input_tensor_name,
        output_tensor_name=output_tensor_name,
    )


async def _tensor_name_hints(
    storage: object, model: str
) -> tuple[str | None, str | None]:
    """Read persisted tensor names when a backend exposes them as metadata."""
    getter = getattr(storage, "get_metadata", None)
    if not callable(getter):
        return "input", "output"
    try:
        metadata_getter = cast("Callable[[str], Awaitable[object]]", getter)
        metadata = await metadata_getter(model)
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        StorageReadError,
        StorageError,
    ):
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
            getter = getattr(metadata, "get_input_tensor_name", None)
            input_name = getter() if callable(getter) else None
        if output_name is None:
            getter = getattr(metadata, "get_output_tensor_name", None)
            output_name = getter() if callable(getter) else None
    try:
        return (
            _text(input_name) if input_name else "input",
            _text(output_name) if output_name else "output",
        )
    except (UnicodeDecodeError, TypeError, ValueError):
        return "input", "output"


def _resolve_loader_options(
    max_background_rows: int | None,
    *,
    include_stored_output: bool | None,
    include_targets: bool | None,
    n_training_rows: int | None,
) -> tuple[int, bool]:
    """Resolve compatibility options and enforce their invariants."""
    if max_background_rows is None:
        max_background_rows = (
            _DEFAULT_TRAINING_ROWS if n_training_rows is None else n_training_rows
        )
    elif n_training_rows is not None and max_background_rows != n_training_rows:
        msg = "max_background_rows and n_training_rows disagree"
        raise LocalDataError(msg)
    if max_background_rows < 1:
        msg = "max_background_rows must be positive"
        raise LocalDataError(msg)
    if include_stored_output is None:
        include_stored_output = bool(include_targets)
    elif include_targets is not None and include_stored_output != include_targets:
        msg = "include_stored_output and include_targets disagree"
        raise LocalDataError(msg)
    return max_background_rows, include_stored_output


async def _load_target_context(
    storage: object, metadata_name: str, input_name: str, prediction_id: str
) -> tuple[int, int, list[str], np.ndarray]:
    """Load and validate the target row and its input aliases."""
    if not await storage.dataset_exists(metadata_name):
        msg = "No stored data exists for the requested model"
        raise LocalDataNotFoundError(msg)
    if not await storage.dataset_exists(input_name):
        msg = "No stored data exists for the requested model"
        raise LocalDataNotFoundError(msg)
    metadata_rows = await storage.dataset_rows(metadata_name)
    input_rows = await storage.dataset_rows(input_name)
    if (
        not isinstance(metadata_rows, int)
        or not isinstance(input_rows, int)
        or metadata_rows < 0
        or input_rows < 0
    ):
        msg = "Stored dataset row counts are invalid"
        raise LocalDataError(msg)
    if metadata_rows != input_rows:
        msg = "Stored input and metadata rows are misaligned"
        raise LocalDataError(msg)
    row_index = await _metadata_matches(
        storage, metadata_name, prediction_id, metadata_rows
    )
    feature_names = [
        _text(name) for name in await storage.get_aliased_column_names(input_name)
    ]
    if not feature_names or len(set(feature_names)) != len(feature_names):
        msg = "Stored input aliases are missing or ambiguous"
        raise LocalDataError(msg)
    instance = _numeric_rows(
        await storage.read_data(input_name, start_row=row_index, n_rows=1),
        name="input",
    )
    if instance.shape[1] != len(feature_names):
        msg = "Stored input columns do not match their aliases"
        raise LocalDataError(msg)
    return input_rows, row_index, feature_names, instance


async def _load_output_context(
    storage: object, output_dataset: str | None, input_rows: int
) -> list[str]:
    """Load and validate stored output aliases when surrogate labels are needed."""
    if output_dataset is None:
        return []
    if not await storage.dataset_exists(output_dataset):
        msg = "Stored output labels are unavailable"
        raise LocalDataError(msg)
    output_rows = await storage.dataset_rows(output_dataset)
    if not isinstance(output_rows, int) or output_rows != input_rows:
        msg = "Stored input and output rows are misaligned"
        raise LocalDataError(msg)
    output_names = [
        _text(name) for name in await storage.get_aliased_column_names(output_dataset)
    ]
    if not output_names or len(set(output_names)) != len(output_names):
        msg = "Stored output aliases are missing or ambiguous"
        raise LocalDataError(msg)
    sample = _numeric_rows(
        await storage.read_data(output_dataset, start_row=0, n_rows=1),
        name="output",
    )
    if sample.shape[1] != len(output_names):
        msg = "Stored output columns do not match their aliases"
        raise LocalDataError(msg)
    return output_names
