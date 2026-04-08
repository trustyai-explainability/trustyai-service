"""
Tests for format detection.

Tests magic byte detection for gzip and JSON formats.
"""

import gzip

import pytest

from src.service.serialization.detection import (
    GZIP_MAGIC,
    detect_format,
    is_gzip,
    is_json,
)


class TestDetectFormat:
    """Test detect_format function."""

    def test_detect_gzip(self):
        """Test detection of gzip format."""
        data = gzip.compress(b"hello world")
        assert detect_format(data) == "gzip"

    def test_detect_json_object(self):
        """Test detection of JSON object."""
        data = b'{"key": "value"}'
        assert detect_format(data) == "json"

    def test_detect_json_array(self):
        """Test detection of JSON array."""
        data = b"[1, 2, 3]"
        assert detect_format(data) == "json"

    def test_detect_json_string(self):
        """Test detection of JSON string."""
        data = b'"hello"'
        assert detect_format(data) == "json"

    def test_detect_json_number(self):
        """Test detection of JSON numbers."""
        assert detect_format(b"123") == "json"
        assert detect_format(b"-45.6") == "json"

    def test_detect_json_boolean(self):
        """Test detection of JSON booleans."""
        assert detect_format(b"true") == "json"
        assert detect_format(b"false") == "json"

    def test_detect_json_null(self):
        """Test detection of JSON null."""
        assert detect_format(b"null") == "json"

    def test_detect_empty_data_raises(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot detect format of empty data"):
            detect_format(b"")

    def test_detect_unknown_format_raises(self):
        """Test that unknown format raises ValueError."""
        data = b"\xaa\xbb\xcc\xdd"  # Random bytes
        with pytest.raises(ValueError, match="Unknown serialization format"):
            detect_format(data)


class TestIsGzip:
    """Test is_gzip helper function."""

    def test_gzip_data(self):
        """Test gzip data detection."""
        data = gzip.compress(b"hello world")
        assert is_gzip(data) is True

    def test_not_gzip(self):
        """Test non-gzip data."""
        data = b'{"key": "value"}'
        assert is_gzip(data) is False


class TestIsJson:
    """Test is_json helper function."""

    def test_json_object(self):
        """Test JSON object detection."""
        data = b'{"key": "value"}'
        assert is_json(data) is True

    def test_json_array(self):
        """Test JSON array detection."""
        data = b"[1, 2, 3]"
        assert is_json(data) is True

    def test_json_string(self):
        """Test JSON string detection."""
        data = b'"hello"'
        assert is_json(data) is True

    def test_json_number(self):
        """Test JSON number detection."""
        assert is_json(b"123") is True
        assert is_json(b"45.6") is True
        assert is_json(b"0") is True

    def test_json_negative_number(self):
        """Test JSON negative number detection."""
        assert is_json(b"-123") is True
        assert is_json(b"-45.6") is True

    def test_json_boolean_true(self):
        """Test JSON boolean true detection."""
        data = b"true"
        assert is_json(data) is True

    def test_json_boolean_false(self):
        """Test JSON boolean false detection."""
        data = b"false"
        assert is_json(data) is True

    def test_json_null(self):
        """Test JSON null detection."""
        data = b"null"
        assert is_json(data) is True

    def test_not_json(self):
        """Test non-JSON data."""
        data = b"hello world"
        assert is_json(data) is False

    def test_empty_data(self):
        """Test empty data."""
        data = b""
        assert is_json(data) is False

    def test_gzip_is_not_json(self):
        """Test gzip data is not detected as JSON."""
        data = gzip.compress(b"hello")
        assert is_json(data) is False


class TestMagicConstants:
    """Test magic byte constants."""

    def test_gzip_magic(self):
        """Test gzip magic bytes are defined."""
        assert GZIP_MAGIC == b"\x1f\x8b"

    def test_actual_gzip_starts_with_magic(self):
        """Test real gzip data starts with magic bytes."""
        data = gzip.compress(b"test")
        assert data.startswith(GZIP_MAGIC)
