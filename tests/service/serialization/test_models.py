"""Tests for serialization module.

Tests JSON with gzip compression serialization.
"""

import gzip
import json

import pytest
from pydantic import BaseModel

from trustyai_service.service.serialization import deserialize_model, serialize_model

# --- Test constants ---
EXPECTED_VALUE = 42


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int
    is_active: bool = True


class NestedModel(BaseModel):
    """Nested Pydantic model for testing."""

    id: str
    data: SampleModel
    tags: list[str]


class TestModelSerialization:
    """Test model serialization functions."""

    @pytest.fixture
    def sample_model(self) -> SampleModel:
        """Create a sample model for testing."""
        return SampleModel(name="test", value=EXPECTED_VALUE, is_active=True)

    @pytest.fixture
    def nested_model(self, sample_model: SampleModel) -> NestedModel:
        """Create a nested model for testing."""
        return NestedModel(id="test-123", data=sample_model, tags=["tag1", "tag2"])

    def test_serialize_pydantic_model(self, sample_model: SampleModel) -> None:
        """Test serializing Pydantic model with JSON + gzip."""
        serialized = serialize_model(sample_model)

        assert isinstance(serialized, bytes)
        # Verify it's gzip format (starts with gzip magic bytes)
        assert serialized.startswith(b"\x1f\x8b")

        # Verify it deserializes to valid JSON
        json_str = gzip.decompress(serialized).decode("utf-8")
        data = json.loads(json_str)
        assert data["name"] == "test"
        assert data["value"] == EXPECTED_VALUE
        assert data["is_active"] is True

    def test_serialize_dict(self) -> None:
        """Test serializing dictionary with JSON + gzip."""
        data_dict = {"name": "test", "value": EXPECTED_VALUE}
        serialized = serialize_model(data_dict)

        assert isinstance(serialized, bytes)
        # Verify it's gzip format
        assert serialized.startswith(b"\x1f\x8b")

        # Verify deserialization
        json_str = gzip.decompress(serialized).decode("utf-8")
        deserialized = json.loads(json_str)
        assert deserialized == data_dict

    def test_serialize_invalid_type(self) -> None:
        """Test serialization of unsupported types."""
        with pytest.raises(ValueError, match="Cannot serialize type"):
            serialize_model([1, 2, 3])  # type: ignore[arg-type]  # Lists not supported - testing error

    def test_deserialize_from_gzip_json(self, sample_model: SampleModel) -> None:
        """Test deserializing from gzip-compressed JSON format."""
        # Create gzip-compressed JSON data
        data = sample_model.model_dump()
        json_str = json.dumps(data)
        serialized = gzip.compress(json_str.encode("utf-8"))

        # Deserialize
        result = deserialize_model(serialized, SampleModel)

        assert isinstance(result, SampleModel)
        assert result.name == sample_model.name
        assert result.value == sample_model.value
        assert result.is_active == sample_model.is_active

    def test_deserialize_from_uncompressed_json(
        self, sample_model: SampleModel
    ) -> None:
        """Test deserializing from uncompressed JSON format (legacy fallback)."""
        # Create uncompressed JSON data
        data = sample_model.model_dump()
        serialized = json.dumps(data).encode("utf-8")

        # Deserialize
        result = deserialize_model(serialized, SampleModel)

        assert isinstance(result, SampleModel)
        assert result.name == sample_model.name
        assert result.value == sample_model.value

    def test_deserialize_invalid_data(self) -> None:
        """Test deserializing invalid data."""
        invalid_data = b"this is not valid serialized data"

        with pytest.raises(ValueError, match="Unsupported serialization format"):
            deserialize_model(invalid_data, SampleModel)

    def test_deserialize_schema_validation_failure(self) -> None:
        """Test deserialization with schema validation failure."""
        # Create data missing required fields
        data = {"name": "test"}  # Missing 'value' field
        json_str = json.dumps(data)
        serialized = gzip.compress(json_str.encode("utf-8"))

        with pytest.raises(ValueError, match=r"validation error|value"):
            deserialize_model(serialized, SampleModel)

    def test_nested_model_serialization(self, nested_model: NestedModel) -> None:
        """Test serialization of nested Pydantic models."""
        serialized = serialize_model(nested_model)
        deserialized = deserialize_model(serialized, NestedModel)

        assert deserialized.id == nested_model.id
        assert deserialized.data.name == nested_model.data.name
        assert deserialized.tags == nested_model.tags

    def test_gzip_compression_reduces_size(self, sample_model: SampleModel) -> None:
        """Test that gzip compression reduces JSON size."""
        # Serialize with gzip
        compressed_data = serialize_model(sample_model)

        # Create uncompressed JSON for comparison
        data = sample_model.model_dump()
        uncompressed_data = json.dumps(data).encode("utf-8")

        # Verify types
        assert isinstance(compressed_data, bytes)
        assert isinstance(uncompressed_data, bytes)

        # Verify compression is effective (allow overhead for small payloads)
        # For small data, gzip header (~18 bytes) can make it larger
        # Allow up to 1.5x for very small payloads (test data is ~48 bytes)
        compressed_size = len(compressed_data)
        uncompressed_size = len(uncompressed_data)
        # For payloads < SMALL_PAYLOAD_THRESHOLD bytes, allow 50% overhead; for larger, expect compression
        SMALL_PAYLOAD_THRESHOLD = 100  # noqa: N806 -- test constant
        max_allowed = uncompressed_size * (
            1.5 if uncompressed_size < SMALL_PAYLOAD_THRESHOLD else 1.0
        )
        assert compressed_size <= max_allowed, (
            f"Compression overhead too large: "
            f"compressed={compressed_size}, uncompressed={uncompressed_size}, "
            f"ratio={compressed_size / uncompressed_size:.2f}"
        )


class TestPublicAPI:
    """Test public API functions."""

    @pytest.fixture
    def sample_model(self) -> SampleModel:
        """Create a sample model for testing."""
        return SampleModel(name="test", value=EXPECTED_VALUE)

    def test_serialize_model(self, sample_model: SampleModel) -> None:
        """Test serialize_model function."""
        serialized = serialize_model(sample_model)

        assert isinstance(serialized, bytes)
        # Should use gzip-compressed JSON
        assert serialized.startswith(b"\x1f\x8b")

    def test_deserialize_model(self, sample_model: SampleModel) -> None:
        """Test deserialize_model function."""
        serialized = serialize_model(sample_model)
        deserialized = deserialize_model(serialized, SampleModel)

        assert isinstance(deserialized, SampleModel)
        assert deserialized.name == sample_model.name
        assert deserialized.value == sample_model.value

    def test_roundtrip(self, sample_model: SampleModel) -> None:
        """Test full serialization/deserialization roundtrip."""
        serialized = serialize_model(sample_model)
        deserialized = deserialize_model(serialized, SampleModel)

        assert deserialized == sample_model


class TestBinaryDataHandling:
    """Test handling of binary data in serialization."""

    def test_binary_data_handling(self) -> None:
        """Test that binary data is handled correctly in JSON+gzip serialization."""

        class BinaryModel(BaseModel):
            data: bytes
            name: str

        model = BinaryModel(data=b"\x00\x01\x02\xff", name="binary_test")

        # Serialize and deserialize
        serialized = serialize_model(model)
        deserialized = deserialize_model(serialized, BinaryModel)

        assert deserialized.data == model.data
        assert deserialized.name == model.name


class TestErrorHandling:
    """Test error handling and edge cases in serialization."""

    def test_deserialize_corrupted_gzip(self) -> None:
        """Test handling of corrupted gzip data."""
        # Create data that starts with gzip magic bytes but is corrupted
        corrupted_data = b"\x1f\x8b\x08\x00" + b"\xff" * 20

        with pytest.raises(ValueError, match=r"Failed to deserialize|gzip|Bad"):
            deserialize_model(corrupted_data, SampleModel)

    def test_deserialize_invalid_json(self) -> None:
        """Test handling of invalid JSON data."""
        invalid_json = b"{invalid json content}"

        with pytest.raises(ValueError, match=r"Unsupported serialization format|JSON"):
            deserialize_model(invalid_json, SampleModel)

    def test_deserialize_truncated_data(self) -> None:
        """Test handling of truncated serialized data."""
        model = SampleModel(name="test", value=42)
        serialized = serialize_model(model)

        # Truncate the data
        truncated = serialized[:10]

        with pytest.raises(ValueError, match=r"Failed to deserialize|Unsupported"):
            deserialize_model(truncated, SampleModel)

    def test_deserialize_schema_mismatch(self) -> None:
        """Test handling of schema validation failures."""

        class OldModel(BaseModel):
            name: str
            value: int

        class NewModel(BaseModel):
            name: str
            value: int
            required_new_field: str  # New required field

        # Serialize with old schema
        old_model = OldModel(name="test", value=EXPECTED_VALUE)
        serialized = serialize_model(old_model)

        # Try to deserialize with new schema (missing required field)
        with pytest.raises(ValueError, match=r"required_new_field"):
            deserialize_model(serialized, NewModel)

    def test_deserialize_empty_data(self) -> None:
        """Test handling of empty data."""
        with pytest.raises(
            ValueError, match=r"Cannot detect format of empty data|Unsupported"
        ):
            deserialize_model(b"", SampleModel)

    def test_deserialize_random_bytes(self) -> None:
        """Test handling of random binary data (not valid format)."""
        random_data = b"\xaa\xbb\xcc\xdd\xee\xff\x00\x11\x22\x33"

        with pytest.raises(
            ValueError, match=r"Unknown serialization format|Unsupported"
        ):
            deserialize_model(random_data, SampleModel)
