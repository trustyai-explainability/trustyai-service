"""Bounded, schema-aware storage loading for local explanations."""

from dataclasses import dataclass, field

import numpy as np

from trustyai_service.service.constants import (
    INPUT_SUFFIX,
    METADATA_ID_COL,
    METADATA_SUFFIX,
    METADATA_TAGS_COL,
    OUTPUT_SUFFIX,
    SYNTHETIC_TAG,
)
from trustyai_service.service.data.storage import get_global_storage_interface

_CHUNK_SIZE = 1000
_MAX_SCAN_ROWS = 1_000_000
_DEFAULT_TRAINING_ROWS = 10_000


@dataclass(frozen=True)
class LocalExplanationData:
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


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def _rows(value: object, *, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Stored {name} data is unavailable")
    array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or not len(array):
        raise ValueError(f"Stored {name} data must be a non-empty matrix")
    return array


def _numeric_rows(value: object, *, name: str) -> np.ndarray:
    array = _rows(value, name=name)
    # Storage values are already typed by the storage backend.  Reject object
    # and string arrays instead of accepting silently coercible values: doing
    # so prevents malformed payloads from becoming valid model features.
    if array.dtype.kind not in "biuf":
        raise ValueError(f"Stored {name} data must be numeric")
    try:
        result = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stored {name} data must be numeric") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"Stored {name} data must be finite")
    return result


def _chunk_matrix(value: object, row_count: int, *, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Stored {name} data is unavailable")
    array = np.asarray(value)
    if array.ndim == 1:
        array = (
            array.reshape(-1, 1) if len(array) == row_count else array.reshape(1, -1)
        )
    if array.ndim != 2 or len(array) != row_count:
        raise ValueError(f"Stored {name} rows are misaligned")
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
                raise ValueError("Stored metadata is missing prediction IDs")
            if _text(row[METADATA_ID_COL]) == prediction_id:
                matches.append(offset + index)
    if len(matches) > 1:
        raise ValueError("Prediction ID is ambiguous")
    if not matches:
        raise ValueError("Prediction ID was not found")
    return matches[0]


async def _load_background(
    storage: object,
    input_dataset: str,
    metadata_dataset: str,
    output_dataset: str | None,
    target_index: int,
    training_rows: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    input_chunks: list[np.ndarray] = []
    output_chunks: list[np.ndarray] = []
    selected = 0
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
    offsets = (
        range(
            ((total_rows - 1) // _CHUNK_SIZE) * _CHUNK_SIZE,
            -1,
            -_CHUNK_SIZE,
        )
        if has_count and total_rows
        else range(0, _MAX_SCAN_ROWS, _CHUNK_SIZE)
    )
    for offset in offsets:
        input_chunk = await storage.read_data(
            input_dataset, start_row=offset, n_rows=_CHUNK_SIZE
        )
        metadata_chunk = await storage.read_data(
            metadata_dataset, start_row=offset, n_rows=_CHUNK_SIZE
        )
        if input_chunk is None or len(input_chunk) == 0:
            continue
        if metadata_chunk is None or len(metadata_chunk) != len(input_chunk):
            raise ValueError("Stored input and metadata rows are misaligned")
        input_matrix = _numeric_rows(
            _chunk_matrix(input_chunk, len(input_chunk), name="input"), name="input"
        )
        output_matrix = None
        if output_dataset is not None:
            output_chunk = await storage.read_data(
                output_dataset, start_row=offset, n_rows=_CHUNK_SIZE
            )
            if output_chunk is None or len(output_chunk) != len(input_chunk):
                raise ValueError("Stored input and output rows are misaligned")
            output_matrix = _numeric_rows(
                _chunk_matrix(output_chunk, len(input_chunk), name="output"),
                name="output",
            )
        selected_indices: list[int] = []
        for index, metadata in enumerate(np.asarray(metadata_chunk)):
            absolute = offset + index
            if absolute == target_index or _synthetic(
                metadata[METADATA_TAGS_COL] if len(metadata) > METADATA_TAGS_COL else []
            ):
                continue
            selected_indices.append(index)
            selected += 1
            if selected >= training_rows:
                break
        if selected_indices:
            input_chunks.append(input_matrix[selected_indices])
            if output_matrix is not None:
                output_chunks.append(output_matrix[selected_indices])
        if selected >= training_rows:
            break
    if not input_chunks:
        raise ValueError("No organic background data is available")
    ordered_inputs = input_chunks[::-1] if has_count else input_chunks
    ordered_outputs = output_chunks[::-1] if has_count else output_chunks
    background = np.concatenate(ordered_inputs, axis=0)
    if output_dataset is None:
        return background, None
    return background, np.concatenate(ordered_outputs, axis=0)


async def load_local_explanation_data(
    model: str,
    prediction_id: str,
    max_background_rows: int | None = None,
    *,
    include_stored_output: bool | None = None,
    include_targets: bool | None = None,
    n_training_rows: int | None = None,
    storage_interface: object | None = None,
) -> LocalExplanationData:
    if max_background_rows is None:
        max_background_rows = (
            _DEFAULT_TRAINING_ROWS if n_training_rows is None else n_training_rows
        )
    elif n_training_rows is not None and max_background_rows != n_training_rows:
        raise ValueError("max_background_rows and n_training_rows disagree")
    if max_background_rows < 1:
        raise ValueError("max_background_rows must be positive")
    if include_stored_output is None:
        include_stored_output = bool(include_targets)
    elif include_targets is not None and include_stored_output != include_targets:
        raise ValueError("include_stored_output and include_targets disagree")
    storage = storage_interface or get_global_storage_interface()
    metadata_name = model + METADATA_SUFFIX
    if not await storage.dataset_exists(metadata_name):
        raise LookupError("No stored data exists for the requested model")

    input_name = model + INPUT_SUFFIX
    metadata_rows = await storage.dataset_rows(metadata_name)
    input_rows = await storage.dataset_rows(input_name)
    if metadata_rows != input_rows:
        raise ValueError("Stored input and metadata rows are misaligned")
    row_index = await _metadata_matches(
        storage, metadata_name, str(prediction_id), metadata_rows
    )
    feature_names = [
        _text(name) for name in await storage.get_aliased_column_names(input_name)
    ]
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Stored input aliases are missing or ambiguous")
    instance = _numeric_rows(
        await storage.read_data(input_name, start_row=row_index, n_rows=1),
        name="input",
    )
    if instance.shape[1] != len(feature_names):
        raise ValueError("Stored input columns do not match their aliases")

    output_dataset = model + OUTPUT_SUFFIX if include_stored_output else None
    output_names: list[str] = []
    if output_dataset is not None:
        if not await storage.dataset_exists(output_dataset):
            raise ValueError("Stored output labels are unavailable")
        if await storage.dataset_rows(output_dataset) != input_rows:
            raise ValueError("Stored input and output rows are misaligned")
        output_names = [
            _text(name)
            for name in await storage.get_aliased_column_names(output_dataset)
        ]
        if not output_names or len(set(output_names)) != len(output_names):
            raise ValueError("Stored output aliases are missing or ambiguous")
    background, targets = await _load_background(
        storage,
        input_name,
        metadata_name,
        output_dataset,
        row_index,
        max_background_rows,
    )
    return LocalExplanationData(
        model_id=model,
        prediction_id=str(prediction_id),
        instance=instance[0],
        feature_names=feature_names,
        background=background,
        background_output=targets,
        output_names=output_names,
        input_tensor_name="input",
        output_tensor_name="output",
    )
