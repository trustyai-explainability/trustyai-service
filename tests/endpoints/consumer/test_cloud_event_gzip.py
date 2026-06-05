"""Tests for Layer 2 gzip decompression in the CloudEvent consumer endpoint.

Knative Eventing strips Content-Encoding headers while leaving the body
gzip-compressed. These tests verify that the CloudEvent endpoint detects
gzip by magic bytes and decompresses before JSON parsing.
"""

import gzip
import json

import pytest

from src.endpoints.consumer.gzip_utils import decompress_if_gzip


class TestDecompressIfGzip:
    """Unit tests for the decompress_if_gzip utility."""

    def test_decompresses_gzip_data(self) -> None:
        """Valid gzip data is decompressed."""
        original = b'{"inputs": [{"name": "x", "shape": [1], "datatype": "FP32", "data": [1.0]}]}'
        compressed = gzip.compress(original)

        result = decompress_if_gzip(compressed)

        assert result == original

    def test_returns_non_gzip_unchanged(self) -> None:
        """Non-gzip data is returned unchanged."""
        data = b'{"inputs": [{"name": "x", "shape": [1], "datatype": "FP32", "data": [1.0]}]}'

        result = decompress_if_gzip(data)

        assert result is data

    def test_returns_empty_bytes_unchanged(self) -> None:
        """Empty bytes are returned unchanged."""
        result = decompress_if_gzip(b"")

        assert result == b""

    def test_returns_single_byte_unchanged(self) -> None:
        """Single byte (too short for magic check) is returned unchanged."""
        result = decompress_if_gzip(b"\x1f")

        assert result == b"\x1f"

    def test_invalid_gzip_with_magic_bytes_returns_original(self) -> None:
        """Data starting with gzip magic but not valid gzip returns original."""
        fake_gzip = b"\x1f\x8b\x00\x00invalid"

        result = decompress_if_gzip(fake_gzip)

        assert result == fake_gzip

    def test_size_limit_raises_on_decompression_bomb(self) -> None:
        """Exceeding max_size raises ValueError."""
        large_data = b"x" * 10_000
        compressed = gzip.compress(large_data)

        with pytest.raises(ValueError, match="exceeds"):
            decompress_if_gzip(compressed, max_size=100)

    def test_size_limit_within_bounds_succeeds(self) -> None:
        """Data within max_size decompresses successfully."""
        data = b'{"test": true}'
        compressed = gzip.compress(data)

        result = decompress_if_gzip(compressed, max_size=1024)

        assert result == data

    def test_preserves_json_fidelity(self) -> None:
        """Decompressed JSON round-trips correctly."""
        payload = {
            "model_name": "example",
            "id": "req-001",
            "outputs": [
                {
                    "name": "predict",
                    "shape": [2, 1],
                    "datatype": "FP64",
                    "data": [[0.1], [0.9]],
                },
            ],
        }
        original = json.dumps(payload).encode()
        compressed = gzip.compress(original)

        result = decompress_if_gzip(compressed)

        assert json.loads(result) == payload
