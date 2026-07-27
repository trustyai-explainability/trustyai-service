"""Tests for format detection.

Tests magic byte detection for gzip and JSON formats.
"""

import gzip

import pytest

from trustyai_service.service.serialization.detection import (
    GZIP_MAGIC,
    detect_format,
    is_gzip,
    is_json,
    safe_gzip_decompress,
)


class TestDetectFormat:
    """Test detect_format function."""

    def test_detect_gzip(self) -> None:
        """Test detection of gzip format."""
        data = gzip.compress(b"hello world")
        assert detect_format(data) == "gzip"

    def test_detect_json_object(self) -> None:
        """Test detection of JSON object."""
        data = b'{"key": "value"}'
        assert detect_format(data) == "json"

    def test_detect_json_array(self) -> None:
        """Test detection of JSON array."""
        data = b"[1, 2, 3]"
        assert detect_format(data) == "json"

    def test_detect_json_string(self) -> None:
        """Test detection of JSON string."""
        data = b'"hello"'
        assert detect_format(data) == "json"

    def test_detect_json_number(self) -> None:
        """Test detection of JSON numbers."""
        assert detect_format(b"123") == "json"
        assert detect_format(b"-45.6") == "json"

    def test_detect_json_boolean(self) -> None:
        """Test detection of JSON booleans."""
        assert detect_format(b"true") == "json"
        assert detect_format(b"false") == "json"

    def test_detect_json_null(self) -> None:
        """Test detection of JSON null."""
        assert detect_format(b"null") == "json"

    def test_detect_empty_data_raises(self) -> None:
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot detect format of empty data"):
            detect_format(b"")

    def test_detect_unknown_format_raises(self) -> None:
        """Test that unknown format raises ValueError."""
        data = b"\xaa\xbb\xcc\xdd"  # Random bytes
        with pytest.raises(ValueError, match="Unknown serialization format"):
            detect_format(data)


class TestIsGzip:
    """Test is_gzip helper function."""

    def test_gzip_data(self) -> None:
        """Test gzip data detection."""
        data = gzip.compress(b"hello world")
        assert is_gzip(data) is True

    def test_not_gzip(self) -> None:
        """Test non-gzip data."""
        data = b'{"key": "value"}'
        assert is_gzip(data) is False


class TestIsJson:
    """Test is_json helper function."""

    def test_json_object(self) -> None:
        """Test JSON object detection."""
        data = b'{"key": "value"}'
        assert is_json(data) is True

    def test_json_array(self) -> None:
        """Test JSON array detection."""
        data = b"[1, 2, 3]"
        assert is_json(data) is True

    def test_json_string(self) -> None:
        """Test JSON string detection."""
        data = b'"hello"'
        assert is_json(data) is True

    def test_json_number(self) -> None:
        """Test JSON number detection."""
        assert is_json(b"123") is True
        assert is_json(b"45.6") is True
        assert is_json(b"0") is True

    def test_json_negative_number(self) -> None:
        """Test JSON negative number detection."""
        assert is_json(b"-123") is True
        assert is_json(b"-45.6") is True

    def test_json_boolean_true(self) -> None:
        """Test JSON boolean true detection."""
        data = b"true"
        assert is_json(data) is True

    def test_json_boolean_false(self) -> None:
        """Test JSON boolean false detection."""
        data = b"false"
        assert is_json(data) is True

    def test_json_null(self) -> None:
        """Test JSON null detection."""
        data = b"null"
        assert is_json(data) is True

    def test_json_with_leading_whitespace(self) -> None:
        """Test JSON with leading whitespace is correctly detected."""
        # JSON spec allows leading whitespace
        assert is_json(b' {"k": 1}') is True
        assert is_json(b'\n\t{"k": 1}') is True
        assert is_json(b"  \n  [1, 2, 3]") is True
        assert is_json(b'\r\n  "hello"') is True
        assert is_json(b"  \t\n  true") is True

    def test_not_json(self) -> None:
        """Test non-JSON data."""
        data = b"hello world"
        assert is_json(data) is False

    def test_empty_data(self) -> None:
        """Test empty data."""
        data = b""
        assert is_json(data) is False

    def test_gzip_is_not_json(self) -> None:
        """Test gzip data is not detected as JSON."""
        data = gzip.compress(b"hello")
        assert is_json(data) is False


class TestSafeGzipDecompress:
    """Test safe_gzip_decompress function."""

    def test_normal_decompression(self) -> None:
        """Test normal gzip decompression works."""
        original = b"hello world"
        compressed = gzip.compress(original)
        assert safe_gzip_decompress(compressed) == original

    def test_decompression_within_limit(self) -> None:
        """Test data within size limit decompresses successfully."""
        original = b"x" * 1000
        compressed = gzip.compress(original)
        assert safe_gzip_decompress(compressed, max_size=2000) == original

    def test_decompression_at_exact_limit(self) -> None:
        """Test data at exact size limit decompresses successfully."""
        original = b"x" * 1000
        compressed = gzip.compress(original)
        assert safe_gzip_decompress(compressed, max_size=1000) == original

    def test_decompression_exceeds_limit(self) -> None:
        """Test that data exceeding size limit raises ValueError."""
        original = b"x" * 1001
        compressed = gzip.compress(original)
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            safe_gzip_decompress(compressed, max_size=1000)

    def test_decompression_bomb_rejected(self) -> None:
        """Test that a high-ratio gzip payload is rejected."""
        # Zeros compress extremely well — this creates a ~100 KB payload
        # that decompresses to 1 MB
        bomb = gzip.compress(b"\x00" * 1_000_000)
        with pytest.raises(ValueError, match="Possible decompression bomb"):
            safe_gzip_decompress(bomb, max_size=500_000)


class TestMagicConstants:
    """Test magic byte constants."""

    def test_gzip_magic(self) -> None:
        """Test gzip magic bytes are defined."""
        assert GZIP_MAGIC == b"\x1f\x8b"

    def test_actual_gzip_starts_with_magic(self) -> None:
        """Test real gzip data starts with magic bytes."""
        data = gzip.compress(b"test")
        assert data.startswith(GZIP_MAGIC)
