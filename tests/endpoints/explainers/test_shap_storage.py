"""Unit tests for ``get_stored_prediction``.

Mocks ``StorageInterface`` to verify model-not-found (404),
prediction-not-found (400), and valid retrieval behaviour.
"""

from http import HTTPStatus
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi import HTTPException

from trustyai_service.endpoints.explainers.local_explainer import get_stored_prediction


def _make_metadata(ids: list[str]) -> np.ndarray:
    """Build a minimal metadata array with id in column 0."""
    rows = [[id_, "2025-01-01T00:00:00", 1.0, []] for id_ in ids]
    return np.array(rows, dtype="O")


class TestGetStoredPrediction:
    """Tests for ``get_stored_prediction``."""

    @pytest.mark.asyncio
    @patch("trustyai_service.endpoints.explainers.local_explainer.storage_interface")
    @patch("trustyai_service.service.data.model_data.get_global_storage_interface")
    async def test_valid_prediction_returns_row(
        self, mock_global: AsyncMock, mock_si: AsyncMock
    ) -> None:
        """Valid prediction ID returns the correct input row and column names."""
        n_features = 3
        row_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        col_names = ["f0", "f1", "f2"]
        metadata = _make_metadata(["pred-1", "pred-2"])

        mock_si.dataset_exists = AsyncMock(return_value=True)
        mock_si.read_data = AsyncMock(
            side_effect=lambda name, *_args, **_kwargs: (
                metadata if "metadata" in name else row_data
            )
        )
        mock_si.get_aliased_column_names = AsyncMock(return_value=col_names)
        mock_global.return_value = mock_si

        row, names = await get_stored_prediction("my-model", "pred-1")

        assert len(row) == n_features
        assert names == col_names

    @pytest.mark.asyncio
    @patch("trustyai_service.endpoints.explainers.local_explainer.storage_interface")
    async def test_unknown_model_raises_404(self, mock_si: AsyncMock) -> None:
        """Missing model metadata dataset raises HTTP 404."""
        mock_si.dataset_exists = AsyncMock(return_value=False)

        with pytest.raises(HTTPException) as exc_info:
            await get_stored_prediction("no-such-model", "pred-x")

        assert exc_info.value.status_code == HTTPStatus.NOT_FOUND

    @pytest.mark.asyncio
    @patch("trustyai_service.endpoints.explainers.local_explainer.storage_interface")
    @patch("trustyai_service.service.data.model_data.get_global_storage_interface")
    async def test_unknown_prediction_id_raises_400(
        self, mock_global: AsyncMock, mock_si: AsyncMock
    ) -> None:
        """Missing prediction ID in stored metadata raises HTTP 400."""
        metadata = _make_metadata(["pred-1", "pred-2"])

        mock_si.dataset_exists = AsyncMock(return_value=True)
        mock_si.read_data = AsyncMock(return_value=metadata)
        mock_global.return_value = mock_si

        with pytest.raises(HTTPException) as exc_info:
            await get_stored_prediction("my-model", "does-not-exist")

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
