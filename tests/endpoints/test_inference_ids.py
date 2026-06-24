"""Tests for GET /info/inference/ids/{model} endpoint."""

from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

FAKE_TIMESTAMP = "2025-01-01T00:00:00"


def _make_metadata(ids: list[str]) -> np.ndarray:
    """Build a metadata array matching the storage layout."""
    rows = [[id_, FAKE_TIMESTAMP, 1735689600.0, []] for id_ in ids]
    return np.array(rows, dtype="O")


def _read_data_side_effect(metadata: np.ndarray):  # noqa: ANN202
    """Return a side_effect callable that returns metadata for metadata datasets."""

    async def _side_effect(
        name: str, *_args: object, **_kwargs: object
    ) -> np.ndarray | None:
        return metadata if "metadata" in name else None

    return _side_effect


class TestGetInferenceIds:
    """Tests for the inference ID retrieval endpoint."""

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.service.data.model_data.get_global_storage_interface")
    def test_returns_ids_for_valid_model(
        self, mock_global_storage: AsyncMock, mock_storage: AsyncMock
    ) -> None:
        """Valid model returns the expected inference IDs."""
        expected_ids = ["req1_0", "req1_1", "req2_0"]
        metadata = _make_metadata(expected_ids)

        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_global_storage.return_value = mock_storage
        mock_storage.read_data = AsyncMock(side_effect=_read_data_side_effect(metadata))

        response = client.get("/info/inference/ids/test-model")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert body["ids"] == [
            {"id": id_, "timestamp": FAKE_TIMESTAMP} for id_ in expected_ids
        ]
        assert body["total"] == 3  # noqa: PLR2004
        assert body["offset"] == 0

    @patch("src.endpoints.metadata.storage_interface")
    def test_returns_404_for_unknown_model(self, mock_storage: AsyncMock) -> None:
        """Unknown model returns 404."""
        mock_storage.dataset_exists = AsyncMock(return_value=False)

        response = client.get("/info/inference/ids/nonexistent")

        assert response.status_code == 404  # noqa: PLR2004
        assert "nonexistent" in response.json()["detail"]

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.service.data.model_data.get_global_storage_interface")
    def test_returns_empty_list_for_model_with_no_data(
        self, mock_global_storage: AsyncMock, mock_storage: AsyncMock
    ) -> None:
        """Model with no recorded inferences returns an empty list."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_global_storage.return_value = mock_storage

        empty = np.array([], dtype="O").reshape(0, 4)
        mock_storage.read_data = AsyncMock(return_value=empty)

        response = client.get("/info/inference/ids/empty-model")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert body["ids"] == []
        assert body["total"] == 0


class TestInferenceIdsPagination:
    """Tests for limit/offset pagination of inference IDs."""

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.service.data.model_data.get_global_storage_interface")
    def test_limit_truncates_results(
        self, mock_global_storage: AsyncMock, mock_storage: AsyncMock
    ) -> None:
        """Limit parameter caps the number of returned IDs."""
        ids = [f"req_{i}" for i in range(10)]
        metadata = _make_metadata(ids)

        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_global_storage.return_value = mock_storage
        mock_storage.read_data = AsyncMock(side_effect=_read_data_side_effect(metadata))

        response = client.get("/info/inference/ids/big-model?limit=3")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert len(body["ids"]) == 3  # noqa: PLR2004
        assert body["ids"] == [
            {"id": id_, "timestamp": FAKE_TIMESTAMP} for id_ in ids[:3]
        ]
        assert body["total"] == 10  # noqa: PLR2004

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.service.data.model_data.get_global_storage_interface")
    def test_offset_skips_results(
        self, mock_global_storage: AsyncMock, mock_storage: AsyncMock
    ) -> None:
        """Offset parameter skips earlier IDs."""
        ids = [f"req_{i}" for i in range(10)]
        metadata = _make_metadata(ids)

        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_global_storage.return_value = mock_storage
        mock_storage.read_data = AsyncMock(side_effect=_read_data_side_effect(metadata))

        response = client.get("/info/inference/ids/big-model?limit=3&offset=7")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert body["ids"] == [
            {"id": id_, "timestamp": FAKE_TIMESTAMP} for id_ in ids[7:10]
        ]
        assert body["total"] == 10  # noqa: PLR2004
        assert body["offset"] == 7  # noqa: PLR2004

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.service.data.model_data.get_global_storage_interface")
    def test_offset_beyond_total_returns_empty(
        self, mock_global_storage: AsyncMock, mock_storage: AsyncMock
    ) -> None:
        """Offset past total count returns empty list with correct total."""
        ids = ["only_one"]
        metadata = _make_metadata(ids)

        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_global_storage.return_value = mock_storage
        mock_storage.read_data = AsyncMock(side_effect=_read_data_side_effect(metadata))

        response = client.get("/info/inference/ids/small-model?offset=100")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert body["ids"] == []
        assert body["total"] == 1

    def test_invalid_limit_rejected(self) -> None:
        """Limit of 0 is rejected with 422."""
        response = client.get("/info/inference/ids/any-model?limit=0")
        assert response.status_code == 422  # noqa: PLR2004

    def test_negative_offset_rejected(self) -> None:
        """Negative offset is rejected with 422."""
        response = client.get("/info/inference/ids/any-model?offset=-1")
        assert response.status_code == 422  # noqa: PLR2004
