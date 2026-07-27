"""Tests for POST /info/names and DELETE /info/names endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from trustyai_service.main import app

client = TestClient(app)


class TestApplyColumnNames:
    """Tests for POST /info/names endpoint."""

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @pytest.mark.asyncio
    async def test_apply_valid_mapping(self, mock_storage: MagicMock) -> None:
        """Valid input/output mappings are applied successfully."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.get_original_column_names = AsyncMock(
            side_effect=lambda ds: ["f1", "f2"] if "inputs" in ds else ["out1"]
        )
        mock_storage.apply_name_mapping = AsyncMock()

        response = client.post(
            "/info/names",
            json={
                "modelId": "test-model",
                "inputMapping": {"f1": "Feature One"},
                "outputMapping": {"out1": "Output One"},
            },
        )

        assert response.status_code == 200  # noqa: PLR2004
        assert "successfully applied" in response.json()["message"]
        assert mock_storage.apply_name_mapping.call_count == 2  # noqa: PLR2004

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @pytest.mark.asyncio
    async def test_apply_invalid_input_column(self, mock_storage: MagicMock) -> None:
        """Mapping with non-existent input column returns 400."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.get_original_column_names = AsyncMock(return_value=["f1", "f2"])

        response = client.post(
            "/info/names",
            json={
                "modelId": "test-model",
                "inputMapping": {"nonexistent": "Friendly"},
            },
        )

        assert response.status_code == 400  # noqa: PLR2004
        assert "No feature found" in response.json()["detail"]
        assert "nonexistent" in response.json()["detail"]
        mock_storage.apply_name_mapping.assert_not_called()

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @pytest.mark.asyncio
    async def test_apply_invalid_output_column(self, mock_storage: MagicMock) -> None:
        """Mapping with non-existent output column returns 400."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.get_original_column_names = AsyncMock(
            side_effect=lambda ds: ["f1"] if "inputs" in ds else ["out1"]
        )

        response = client.post(
            "/info/names",
            json={
                "modelId": "test-model",
                "outputMapping": {"bad_col": "Friendly"},
            },
        )

        assert response.status_code == 400  # noqa: PLR2004
        assert "No output found" in response.json()["detail"]
        assert "bad_col" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @pytest.mark.asyncio
    async def test_apply_model_not_found(self, mock_storage: MagicMock) -> None:
        """Mapping for unknown model returns 400."""
        mock_storage.dataset_exists = AsyncMock(return_value=False)

        response = client.post(
            "/info/names",
            json={
                "modelId": "unknown-model",
                "inputMapping": {"f1": "Friendly"},
            },
        )

        assert response.status_code == 400  # noqa: PLR2004
        assert "No metadata found" in response.json()["detail"]


class TestRemoveColumnNames:
    """Tests for DELETE /info/names endpoint."""

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @pytest.mark.asyncio
    async def test_delete_mapping_success(self, mock_storage: MagicMock) -> None:
        """Clearing name mappings succeeds with plain string body."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.clear_name_mapping = AsyncMock()

        response = client.request(
            "DELETE",
            "/info/names",
            content='"test-model"',
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200  # noqa: PLR2004
        assert "successfully cleared" in response.json()["message"]
        assert mock_storage.clear_name_mapping.call_count == 2  # noqa: PLR2004

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, mock_storage: MagicMock) -> None:
        """Clearing mappings for unknown model returns 400."""
        mock_storage.dataset_exists = AsyncMock(return_value=False)

        response = client.request(
            "DELETE",
            "/info/names",
            content='"unknown-model"',
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400  # noqa: PLR2004
        assert "No metadata found" in response.json()["detail"]
