"""Regression tests for local explanation storage semantics."""

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
    ) -> None:
        self.metadata = metadata
        self.inputs = inputs
        self.outputs = outputs

    async def dataset_exists(self, name: str) -> bool:
        return self.outputs is not None or not name.endswith("_outputs")

    async def dataset_rows(self, _name: str) -> int:
        return len(self.inputs)

    async def get_aliased_column_names(self, name: str) -> list[str]:
        return ["f0", "f1"] if name.endswith("_inputs") else ["score"]

    async def read_data(
        self, name: str, start_row: int = 0, n_rows: int | None = None
    ) -> np.ndarray:
        values = self.metadata if name.endswith("_metadata") else self.inputs
        if name.endswith("_outputs"):
            if self.outputs is None:
                raise ValueError(_MISSING_OUTPUTS)
            values = self.outputs
        end = None if n_rows is None else start_row + n_rows
        return values[start_row:end]


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
