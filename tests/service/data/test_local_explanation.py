"""Regression tests for local explanation storage semantics."""

from typing import cast

import numpy as np
import pytest

from trustyai_service.service.data.local_explanation import load_local_explanation_data

_MISSING_OUTPUTS = "missing outputs"


class _Storage:
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
        self.output_aliases = cast(
            "list[str] | None", options.get("output_aliases")
        ) or ["score"]
        self.collapse_single_row = bool(options.get("collapse_single_row", False))
        self.tensor_names = cast("dict[str, str] | None", options.get("tensor_names"))

    async def dataset_exists(self, name: str) -> bool:
        return self.outputs is not None or not name.endswith("_outputs")

    async def dataset_rows(self, _name: str) -> int:
        return len(self.inputs)

    async def get_aliased_column_names(self, name: str) -> list[str]:
        return ["f0", "f1"] if name.endswith("_inputs") else list(self.output_aliases)

    async def get_metadata(self, _model: str) -> dict[str, str] | None:
        return self.tensor_names

    async def read_data(
        self, name: str, start_row: int = 0, n_rows: int | None = None
    ) -> np.ndarray:
        values = self.metadata if name.endswith("_metadata") else self.inputs
        if name.endswith("_outputs"):
            if self.outputs is None:
                raise ValueError(_MISSING_OUTPUTS)
            values = self.outputs
        end = None if n_rows is None else start_row + n_rows
        result = values[start_row:end]
        if self.collapse_single_row and len(result) == 1:
            return result[0]
        return result


def _metadata() -> np.ndarray:
    return np.array(
        [
            ["target", "t", 0, []],
            ["organic", "t", 0, []],
            ["synthetic", "t", 0, "_trustyai_synthetic"],
        ],
        dtype=object,
    )


@pytest.mark.asyncio
async def test_loader_excludes_target_and_synthetic_rows() -> None:
    """Exclude the target and synthetic rows from the background population."""
    storage = _Storage(_metadata(), np.array([[9, 9], [1, 2], [3, 4]], dtype=float))
    data = await load_local_explanation_data(
        "m",
        "target",
        include_targets=False,
        n_training_rows=10,
        storage_interface=storage,
    )
    assert data.instance.tolist() == [9, 9]
    assert data.background.tolist() == [[1, 2]]


@pytest.mark.asyncio
async def test_loader_rejects_duplicate_prediction_ids() -> None:
    """Reject ambiguous storage when more than one row matches the target ID."""
    metadata = _metadata()
    metadata[1, 0] = "target"
    storage = _Storage(metadata, np.ones((3, 2)))
    with pytest.raises(ValueError, match="ambiguous"):
        await load_local_explanation_data(
            "m", "target", include_targets=False, storage_interface=storage
        )


@pytest.mark.asyncio
async def test_model_mode_does_not_require_outputs() -> None:
    """Allow real-model loading when stored prediction outputs are absent."""
    storage = _Storage(_metadata(), np.ones((3, 2)))
    data = await load_local_explanation_data(
        "m", "target", include_targets=False, storage_interface=storage
    )
    assert data.background_output is None


@pytest.mark.asyncio
async def test_surrogate_mode_requires_stored_outputs() -> None:
    """Require stored outputs when loading data for surrogate execution."""
    storage = _Storage(_metadata(), np.ones((3, 2)))
    with pytest.raises(ValueError, match="output labels"):
        await load_local_explanation_data(
            "m", "target", include_stored_output=True, storage_interface=storage
        )


@pytest.mark.asyncio
async def test_loader_rejects_string_features() -> None:
    """Reject string-valued features instead of silently coercing them."""
    storage = _Storage(_metadata(), np.array([["1", "2"], ["3", "4"], ["5", "6"]]))
    with pytest.raises(ValueError, match="must be numeric"):
        await load_local_explanation_data(
            "m", "target", include_stored_output=False, storage_interface=storage
        )


@pytest.mark.asyncio
async def test_loader_scans_non_aligned_storage_chunks_without_gaps() -> None:
    """Scan reverse storage chunks without skipping rows at chunk boundaries."""
    rows = 2501
    metadata = np.empty((rows, 4), dtype=object)
    metadata[:, 0] = ["target" if index == 0 else f"p-{index}" for index in range(rows)]
    metadata[:, 1] = "t"
    metadata[:, 2] = 0
    metadata[:, 3] = [[] for _ in range(rows)]
    inputs = np.column_stack((np.arange(rows, dtype=float), np.ones(rows)))
    storage = _Storage(metadata, inputs)
    data = await load_local_explanation_data(
        "m", "target", 2500, include_stored_output=False, storage_interface=storage
    )
    assert len(data.background) == 2500
    assert data.background[0, 0] == 1
    assert data.background[-1, 0] == 2500


@pytest.mark.asyncio
async def test_loader_restores_one_row_reads_to_matrices() -> None:
    """Handle storage backends that collapse a one-row chunk to one dimension."""
    rows = 1001
    metadata = np.empty((rows, 4), dtype=object)
    metadata[:, 0] = ["target" if index == 0 else f"p-{index}" for index in range(rows)]
    metadata[:, 1] = "t"
    metadata[:, 2] = 0
    metadata[:, 3] = [[] for _ in range(rows)]
    inputs = np.zeros((rows, 2), dtype=float)
    inputs[-1] = [7.0, 8.0]
    outputs = np.zeros((rows, 1), dtype=float)
    outputs[-1] = 3.0
    storage = _Storage(
        metadata,
        inputs,
        outputs,
        collapse_single_row=True,
    )

    data = await load_local_explanation_data(
        "m",
        "target",
        1,
        include_stored_output=True,
        storage_interface=storage,
    )

    assert data.background.tolist() == [[7.0, 8.0]]
    assert data.background_output is not None
    assert data.background_output.tolist() == [[3.0]]


@pytest.mark.asyncio
async def test_loader_rejects_output_alias_width_mismatch() -> None:
    """Reject aliases that do not describe the stored output matrix width."""
    storage = _Storage(
        _metadata(),
        np.ones((3, 2)),
        np.ones((3, 1)),
        output_aliases=["score", "probability"],
    )
    with pytest.raises(ValueError, match="columns"):
        await load_local_explanation_data(
            "m", "target", include_stored_output=True, storage_interface=storage
        )


@pytest.mark.asyncio
async def test_loader_preserves_storage_tensor_name_hints() -> None:
    """Expose persisted tensor names as hints without overriding KServe metadata."""
    storage = _Storage(
        _metadata(),
        np.ones((3, 2)),
        tensor_names={"inputTensorName": "features", "outputTensorName": "score"},
    )
    data = await load_local_explanation_data(
        "m", "target", include_stored_output=False, storage_interface=storage
    )
    assert data.input_tensor_name == "features"
    assert data.output_tensor_name == "score"
