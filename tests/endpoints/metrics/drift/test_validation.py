"""Tests for shared drift validation utilities."""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from trustyai_service.endpoints.metrics.drift.validation import validate_drift_request


class MockDriftRequest(BaseModel):
    """Mock drift request for testing."""

    model_id: str
    reference_tag: str | None = None
    fit_columns: list[str] | None = None


@pytest.mark.asyncio
class TestValidateDriftRequest:
    """Tests for validate_drift_request() function."""

    async def test_validate_missing_reference_tag_raises_400(self) -> None:
        """Test that missing referenceTag raises HTTP 400."""
        request = MockDriftRequest(
            model_id="test-model",
            reference_tag=None,
            fit_columns=["feature1"],
        )

        with pytest.raises(HTTPException) as exc_info:
            await validate_drift_request(request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
        assert "referenceTag is required" in exc_info.value.detail

    async def test_validate_empty_reference_tag_raises_400(self) -> None:
        """Test that empty referenceTag raises HTTP 400."""
        request = MockDriftRequest(
            model_id="test-model",
            reference_tag="",
            fit_columns=["feature1"],
        )

        with pytest.raises(HTTPException) as exc_info:
            await validate_drift_request(request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
        assert "referenceTag is required" in exc_info.value.detail

    async def test_validate_whitespace_reference_tag_raises_400(self) -> None:
        """Test that whitespace-only referenceTag raises HTTP 400."""
        request = MockDriftRequest(
            model_id="test-model",
            reference_tag="   ",
            fit_columns=["feature1"],
        )

        with pytest.raises(HTTPException) as exc_info:
            await validate_drift_request(request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
        assert "referenceTag is required" in exc_info.value.detail

    async def test_validate_omitted_fit_columns_derives_from_metadata(self) -> None:
        """Test that omitted fitColumns auto-derives from metadata."""
        request = MockDriftRequest(
            model_id="test-model",
            reference_tag="baseline",
            fit_columns=None,  # Omitted
        )

        # Mock get_shared_data_source
        mock_metadata = MagicMock()
        mock_metadata.input_schema.items.keys.return_value = ["feature1", "feature2"]
        mock_data_source = MagicMock()
        mock_data_source.get_metadata = AsyncMock(return_value=mock_metadata)

        with patch(
            "trustyai_service.endpoints.metrics.drift.validation.get_shared_data_source",
            return_value=mock_data_source,
        ):
            result = await validate_drift_request(request)

        # Should auto-derive from metadata
        assert result == ["feature1", "feature2"]
        assert request.fit_columns == ["feature1", "feature2"]
        mock_data_source.get_metadata.assert_called_once_with("test-model")

    async def test_validate_explicit_empty_fit_columns_raises_400(self) -> None:
        """Test that explicit empty fitColumns raises HTTP 400."""
        request = MockDriftRequest(
            model_id="test-model",
            reference_tag="baseline",
            fit_columns=[],  # Explicit empty
        )

        with pytest.raises(HTTPException) as exc_info:
            await validate_drift_request(request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
        assert (
            "fitColumns must contain at least one non-empty feature name"
            in exc_info.value.detail
        )

    async def test_validate_provided_fit_columns_strips_whitespace(self) -> None:
        """Test that provided fitColumns are stripped of whitespace."""
        request = MockDriftRequest(
            model_id="test-model",
            reference_tag="baseline",
            fit_columns=["  feature1  ", "feature2", "  feature3"],
        )

        result = await validate_drift_request(request)

        # Should strip whitespace
        assert result == ["feature1", "feature2", "feature3"]
        assert request.fit_columns == ["feature1", "feature2", "feature3"]

    async def test_validate_whitespace_only_fit_columns_raises_400(self) -> None:
        """Test that whitespace-only fitColumns raises HTTP 400."""
        request = MockDriftRequest(
            model_id="test-model",
            reference_tag="baseline",
            fit_columns=["  ", "", "   "],
        )

        with pytest.raises(HTTPException) as exc_info:
            await validate_drift_request(request)

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
        assert (
            "fitColumns must contain at least one non-empty feature name"
            in exc_info.value.detail
        )

    async def test_validate_valid_request_succeeds(self) -> None:
        """Test that valid request with referenceTag and fitColumns succeeds."""
        request = MockDriftRequest(
            model_id="test-model",
            reference_tag="baseline",
            fit_columns=["feature1", "feature2"],
        )

        result = await validate_drift_request(request)

        assert result == ["feature1", "feature2"]
        assert request.fit_columns == ["feature1", "feature2"]
