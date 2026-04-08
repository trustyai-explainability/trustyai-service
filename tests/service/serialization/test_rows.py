"""
Tests for row serialization.

Tests serialize_rows/deserialize_rows with JSON + gzip.
"""

import gzip
import json

import numpy as np
import pytest

from src.service.data.storage.pvc import MAX_VOID_TYPE_LENGTH
from src.service.serialization.encoders import json_encoder
from src.service.serialization.rows import deserialize_rows, serialize_rows


class TestSerializeRows:
    """Test serialize_rows function."""

    def test_serialize_simple_rows(self):
        """Test serializing simple rows."""
        rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        serialized = serialize_rows(rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)

        assert isinstance(serialized, np.ndarray)
        assert len(serialized) == 3
        # Check dtype is void
        assert serialized.dtype.kind == "V"

    def test_serialize_numpy_array(self):
        """Test serializing numpy array (converted to list)."""
        rows = np.array([[1, 2], [3, 4], [5, 6]]).tolist()
        serialized = serialize_rows(rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)

        assert isinstance(serialized, np.ndarray)
        assert len(serialized) == 3

    def test_serialize_mixed_types(self):
        """Test serializing rows with mixed types."""
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
        serialized = serialize_rows(rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)

        assert isinstance(serialized, np.ndarray)
        assert len(serialized) == 3

    def test_serialize_exceeds_max_size(self):
        """Test that exceeding max void type length raises error."""
        # Create large row that exceeds limit
        large_row = [list(range(1000))]

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            serialize_rows(large_row, max_void_type_length=10)

    def test_serialize_empty_list(self):
        """Test serializing empty list."""
        serialized = serialize_rows([], max_void_type_length=MAX_VOID_TYPE_LENGTH)

        assert isinstance(serialized, np.ndarray)
        assert len(serialized) == 0

    def test_dynamic_void_size(self):
        """Test that void dtype size matches largest row."""
        rows = [[1, 2], [3, 4, 5, 6]]  # Second row is larger
        serialized = serialize_rows(rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)

        # Get the size of serialized data (JSON + gzip)
        row_sizes = [len(gzip.compress(json.dumps(row, default=json_encoder).encode("utf-8"))) for row in rows]
        max_size = max(row_sizes)

        # Verify dtype size matches largest row
        assert serialized.dtype.itemsize == max_size

    def test_serialized_data_is_gzip(self):
        """Test that serialized data uses gzip compression."""
        rows = [[1, 2, 3]]
        serialized = serialize_rows(rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)

        # First row should start with gzip magic bytes
        first_row = bytes(serialized[0])
        assert first_row.startswith(b"\x1f\x8b")


class TestDeserializeRows:
    """Test deserialize_rows function."""

    def test_deserialize_json_gzip_simple(self):
        """Test deserializing JSON + gzip serialized rows."""
        original_rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        serialized = serialize_rows(original_rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)
        deserialized = deserialize_rows(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert len(deserialized) == 3
        assert deserialized.dtype == object

        # Verify data integrity
        for i, row in enumerate(deserialized):
            assert list(row) == original_rows[i]

    def test_deserialize_json_gzip_mixed_types(self):
        """Test deserializing rows with mixed types."""
        original_rows = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        serialized = serialize_rows(original_rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)
        deserialized = deserialize_rows(serialized)

        # Verify data integrity
        for i, row in enumerate(deserialized):
            assert row == original_rows[i]

    def test_deserialize_with_padding(self):
        """Test deserializing JSON + gzip data with padding (fixed-size void arrays)."""
        original_rows = [[1], [2, 3, 4]]  # Different sizes

        # Serialize with JSON + gzip
        serialized = serialize_rows(original_rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)

        # Deserialize (should handle padding correctly)
        deserialized = deserialize_rows(serialized)

        # Verify data integrity
        assert list(deserialized[0]) == [1]
        assert list(deserialized[1]) == [2, 3, 4]

    def test_invalid_gzip_json_data(self):
        """Test handling of invalid gzip/JSON data."""
        # Create data with gzip magic bytes but invalid compressed content
        invalid_data = b"\x1f\x8b\x00\x00invalid"  # gzip magic + invalid data
        void_dtype = f"V{len(invalid_data)}"
        serialized = np.array([np.void(invalid_data)], dtype=void_dtype)

        with pytest.raises(ValueError, match="Failed to deserialize row as gzip-compressed JSON"):
            deserialize_rows(serialized)

    def test_roundtrip_preserves_data_types(self):
        """Test that roundtrip preserves various data types."""
        original_rows = [
            [1, 2, 3],  # integers
            [1.5, 2.7, 3.9],  # floats
            ["a", "b", "c"],  # strings
            [True, False, True],  # booleans
            [None, None, None],  # None values
        ]

        serialized = serialize_rows(original_rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)
        deserialized = deserialize_rows(serialized)

        for i, row in enumerate(deserialized):
            assert list(row) == original_rows[i]


class TestDecompressionSafety:
    """Test protection against malicious or problematic compressed data."""

    def test_serialize_rejects_oversized_row(self):
        """Test that serialization rejects rows that would exceed the limit."""
        import random

        # Create random data that won't compress well
        random.seed(42)
        max_size = 512
        # Create data that will exceed limit even after compression
        # Random data doesn't compress well, so we need 2-3x the limit
        oversized_data = "".join(chr(random.randint(32, 126)) for _ in range(max_size * 3))
        rows = [[oversized_data]]

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            serialize_rows(rows, max_void_type_length=max_size)

    def test_deserialize_corrupted_gzip_row(self):
        """Test handling of corrupted gzip data in rows."""
        # Create a row with corrupted gzip data (starts with gzip magic but invalid)
        corrupted = b"\x1f\x8b\x08\x00\xff\xff\xff\xff"  # Invalid gzip
        void_array = np.array([np.void(corrupted)], dtype=f"V{len(corrupted)}")

        with pytest.raises(ValueError, match="Failed to deserialize row as gzip"):
            deserialize_rows(void_array)

    def test_deserialize_non_gzip_row_rejected(self):
        """Test that non-gzip rows are properly rejected."""
        # Create data that isn't gzip
        non_gzip_data = b"this is not gzip data"
        void_array = np.array([np.void(non_gzip_data)], dtype=f"V{len(non_gzip_data)}")

        with pytest.raises(ValueError, match="Unsupported serialization format|Expected gzip"):
            deserialize_rows(void_array)

    def test_large_but_valid_rows(self):
        """Test that large but legitimate rows work correctly."""
        # Create legitimately large rows (but within limits)
        large_row = [list(range(1000))]  # 1000 integers
        serialized = serialize_rows(large_row, max_void_type_length=MAX_VOID_TYPE_LENGTH)
        deserialized = deserialize_rows(serialized)

        assert len(deserialized) == 1
        assert list(deserialized[0]) == large_row[0]

    def test_mixed_row_sizes(self):
        """Test handling of rows with widely varying sizes."""
        rows = [
            [1, 2, 3],  # Small
            [{"key": "value"} for _ in range(50)],  # Medium
            [list(range(100))],  # Large
        ]

        serialized = serialize_rows(rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)
        deserialized = deserialize_rows(serialized)

        assert len(deserialized) == 3
        assert list(deserialized[0]) == rows[0]
        assert deserialized[1] == rows[1]
        assert deserialized[2] == rows[2]

    def test_compression_effectiveness(self):
        """Test that compression actually reduces size for repetitive data."""
        import json

        # Highly repetitive data (compresses well)
        repetitive_row = [[1, 1, 1, 1, 1] * 100]
        json_size = len(json.dumps(repetitive_row[0]).encode("utf-8"))

        serialized = serialize_rows(repetitive_row, max_void_type_length=MAX_VOID_TYPE_LENGTH)

        # Compressed size should be significantly smaller than JSON
        compressed_size = len(bytes(serialized[0]))
        assert compressed_size < json_size * 0.5  # At least 50% compression
