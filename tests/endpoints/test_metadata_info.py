"""Tests for /info endpoint schema fields."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.endpoints.metadata import _build_readable_schema
from src.main import app
from src.service.payloads.service.schema import Schema
from src.service.payloads.service.schema_item import SchemaItem
from src.service.payloads.values.data_type import DataType

client = TestClient(app)


class TestBuildReadableSchema:
    """Tests for _build_readable_schema helper."""

    def test_converts_schema_with_items(self) -> None:
        """Schema items are serialized with type and index."""
        schema = Schema(
            items={
                "col1": SchemaItem(DataType.DOUBLE, "col1", 0),
                "col2": SchemaItem(DataType.INT32, "col2", 1),
            },
        )
        result = _build_readable_schema(schema)

        assert result["items"]["col1"] == {"type": "DOUBLE", "index": 0}
        assert result["items"]["col2"] == {"type": "INT32", "index": 1}
        assert result["nameMapping"] == {}

    def test_includes_name_mapping(self) -> None:
        """Name mappings are included in the output."""
        schema = Schema(
            items={"col1": SchemaItem(DataType.DOUBLE, "col1", 0)},
            name_mapping={"col1": "Friendly Name"},
        )
        result = _build_readable_schema(schema)

        assert result["nameMapping"] == {"col1": "Friendly Name"}

    def test_empty_schema(self) -> None:
        """Empty schema returns empty items and nameMapping."""
        schema = Schema()
        result = _build_readable_schema(schema)

        assert result == {"items": {}, "nameMapping": {}}


class TestInfoEndpointSchema:
    """Tests for /info endpoint inputSchema/outputSchema fields."""

    @patch("src.endpoints.metadata.get_data_source")
    @pytest.mark.asyncio
    async def test_info_includes_schemas(self, mock_get_ds: MagicMock) -> None:
        """The /info response includes inputSchema and outputSchema."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})

        mock_metadata = MagicMock()
        mock_metadata.input_tensor_name = "input"
        mock_metadata.output_tensor_name = "output"
        mock_metadata.input_schema = Schema(
            items={"f1": SchemaItem(DataType.DOUBLE, "f1", 0)},
        )
        mock_metadata.output_schema = Schema(
            items={"out": SchemaItem(DataType.DOUBLE, "out", 0)},
        )
        mock_ds.get_metadata = AsyncMock(return_value=mock_metadata)
        mock_ds.get_num_observations = AsyncMock(return_value=100)
        mock_ds.has_recorded_inferences = AsyncMock(return_value=True)
        mock_get_ds.return_value = mock_ds

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        data = response.json()["test-model"]["data"]
        assert "inputSchema" in data
        assert "outputSchema" in data
        assert data["inputSchema"]["items"]["f1"] == {"type": "DOUBLE", "index": 0}
        assert data["outputSchema"]["items"]["out"] == {"type": "DOUBLE", "index": 0}
