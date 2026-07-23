"""Tests for JSON encoders and decoders.

Tests custom encoding/decoding for numpy arrays, scalars, binary data, and datetimes.
"""

import base64
import datetime
import json

import numpy as np
import pytest

from trustyai_service.service.serialization.encoders import (
    json_decoder_hook,
    json_encoder,
)

# --- Test constants ---
EXPECTED_INT = 42
EXPECTED_FLOAT = 3.14


class TestJsonEncoder:
    """Test json_encoder function."""

    def test_encode_numpy_array(self) -> None:
        """Test encoding numpy arrays to lists."""
        arr = np.array([1, 2, 3, 4, 5])
        result = json_encoder(arr)

        assert result == [1, 2, 3, 4, 5]
        assert isinstance(result, list)

    def test_encode_numpy_2d_array(self) -> None:
        """Test encoding 2D numpy arrays."""
        arr = np.array([[1, 2], [3, 4]])
        result = json_encoder(arr)

        assert result == [[1, 2], [3, 4]]

    def test_encode_numpy_integer(self) -> None:
        """Test encoding numpy integer scalars."""
        value = np.int64(EXPECTED_INT)
        result = json_encoder(value)

        assert result == EXPECTED_INT
        assert isinstance(result, int)

    def test_encode_numpy_float(self) -> None:
        """Test encoding numpy float scalars."""
        value = np.float64(EXPECTED_FLOAT)
        result = json_encoder(value)

        assert result == EXPECTED_FLOAT
        assert isinstance(result, float)

    def test_encode_numpy_bool(self) -> None:
        """Test encoding numpy bool scalars."""
        assert json_encoder(np.bool_(True)) is True  # noqa: FBT003
        assert json_encoder(np.bool_(False)) is False  # noqa: FBT003
        assert isinstance(json_encoder(np.bool_(True)), bool)  # noqa: FBT003

    def test_encode_datetime(self) -> None:
        """Test encoding datetime objects."""
        dt = datetime.datetime(2026, 1, 15, 10, 30, 0, tzinfo=datetime.UTC)
        result = json_encoder(dt)

        assert result == {"__type__": "datetime", "data": dt.isoformat()}

    def test_encode_date(self) -> None:
        """Test encoding date objects."""
        d = datetime.date(2026, 1, 15)
        result = json_encoder(d)

        assert result == {"__type__": "date", "data": "2026-01-15"}

    def test_encode_bytes_ascii(self) -> None:
        """Test encoding ASCII bytes with base64."""
        data = b"hello"
        result = json_encoder(data)

        assert result == {
            "__type__": "bytes",
            "data": base64.b64encode(b"hello").decode("ascii"),
        }

    def test_encode_bytes_binary(self) -> None:
        """Test encoding binary data with base64."""
        data = b"\x00\x01\x02\xff"
        result = json_encoder(data)

        assert result == {"__type__": "bytes", "data": "AAEC/w=="}

    def test_encode_unsupported_type_raises(self) -> None:
        """Test that unsupported types raise TypeError instead of silent conversion."""

        class CustomObject:
            pass

        with pytest.raises(TypeError, match="CustomObject"):
            json_encoder(CustomObject())

    def test_encode_in_json_dumps(self) -> None:
        """Test encoder works with json.dumps()."""
        data = {
            "array": np.array([1, 2, 3]),
            "scalar": np.int64(EXPECTED_INT),
            "bytes": b"\x00\xff",
        }

        json_str = json.dumps(data, default=json_encoder)
        result = json.loads(json_str)

        assert result["array"] == [1, 2, 3]
        assert result["scalar"] == EXPECTED_INT
        assert result["bytes"]["__type__"] == "bytes"


class TestJsonDecoderHook:
    """Test json_decoder_hook function."""

    def test_decode_bytes(self) -> None:
        """Test decoding base64-encoded bytes."""
        encoded = {"__type__": "bytes", "data": "AAEC/w=="}
        result = json_decoder_hook(encoded)

        assert result == b"\x00\x01\x02\xff"
        assert isinstance(result, bytes)

    def test_decode_regular_dict(self) -> None:
        """Test that regular dicts pass through unchanged."""
        data = {"key": "value", "number": 42}
        result = json_decoder_hook(data)

        assert result == data

    def test_decode_datetime(self) -> None:
        """Test decoding ISO 8601 datetime strings."""
        encoded = {"__type__": "datetime", "data": "2026-01-15T10:30:00+00:00"}
        result = json_decoder_hook(encoded)

        assert isinstance(result, datetime.datetime)
        assert result == datetime.datetime(2026, 1, 15, 10, 30, 0, tzinfo=datetime.UTC)

    def test_decode_date(self) -> None:
        """Test decoding ISO 8601 date strings."""
        encoded = {"__type__": "date", "data": "2026-01-15"}
        result = json_decoder_hook(encoded)

        assert isinstance(result, datetime.date)
        assert result == datetime.date(2026, 1, 15)

    def test_decode_dict_with_type_but_not_bytes(self) -> None:
        """Test dict with unknown __type__ passes through unchanged."""
        data = {"__type__": "other", "data": "something"}
        result = json_decoder_hook(data)

        assert result == data

    def test_decode_in_json_loads(self) -> None:
        """Test decoder hook works with json.loads()."""
        json_str = '{"data": {"__type__": "bytes", "data": "AAEC/w=="}}'
        result = json.loads(json_str, object_hook=json_decoder_hook)

        assert result["data"] == b"\x00\x01\x02\xff"

    def test_decode_bytes_missing_data_raises(self) -> None:
        """Test that bytes dict missing data field raises ValueError."""
        encoded = {"__type__": "bytes"}

        with pytest.raises(ValueError, match="missing 'data' field"):
            json_decoder_hook(encoded)

    def test_decode_datetime_missing_data_raises(self) -> None:
        """Test that datetime dict missing data field raises ValueError."""
        encoded = {"__type__": "datetime"}

        with pytest.raises(ValueError, match="missing 'data' field"):
            json_decoder_hook(encoded)

    def test_decode_date_missing_data_raises(self) -> None:
        """Test that date dict missing data field raises ValueError."""
        encoded = {"__type__": "date"}

        with pytest.raises(ValueError, match="missing 'data' field"):
            json_decoder_hook(encoded)

    def test_decode_bytes_invalid_base64_raises(self) -> None:
        """Test that invalid base64 in bytes raises ValueError."""
        encoded = {"__type__": "bytes", "data": "!!invalid!!"}

        with pytest.raises(ValueError, match="Invalid base64"):
            json_decoder_hook(encoded)

    def test_decode_datetime_invalid_format_raises(self) -> None:
        """Test that invalid datetime format raises ValueError."""
        encoded = {"__type__": "datetime", "data": "not-a-datetime"}

        with pytest.raises(ValueError, match="Invalid ISO datetime"):
            json_decoder_hook(encoded)

    def test_decode_date_invalid_format_raises(self) -> None:
        """Test that invalid date format raises ValueError."""
        encoded = {"__type__": "date", "data": "not-a-date"}

        with pytest.raises(ValueError, match="Invalid ISO date"):
            json_decoder_hook(encoded)


class TestEncoderDecoderRoundtrip:
    """Test encoding and decoding work together."""

    def test_roundtrip_bytes(self) -> None:
        """Test bytes survive encode/decode roundtrip."""
        original = b"\x00\x01\x02\xff\xfe"

        # Encode
        encoded = json_encoder(original)
        json_str = json.dumps(encoded)

        # Decode
        decoded_obj = json.loads(json_str, object_hook=json_decoder_hook)

        assert decoded_obj == original

    def test_roundtrip_datetime(self) -> None:
        """Test datetime survives encode/decode roundtrip."""
        original = datetime.datetime(2026, 1, 15, 10, 30, 0, tzinfo=datetime.UTC)

        encoded = json_encoder(original)
        json_str = json.dumps(encoded)
        decoded = json.loads(json_str, object_hook=json_decoder_hook)

        assert decoded == original

    def test_roundtrip_date(self) -> None:
        """Test date survives encode/decode roundtrip."""
        original = datetime.date(2026, 1, 15)

        encoded = json_encoder(original)
        json_str = json.dumps(encoded)
        decoded = json.loads(json_str, object_hook=json_decoder_hook)

        assert decoded == original

    def test_roundtrip_complex_structure(self) -> None:
        """Test complex nested structure with mixed types."""
        original = {
            "arrays": [np.array([1, 2, 3]), np.array([4, 5])],
            "scalars": [np.int64(EXPECTED_INT), np.float64(EXPECTED_FLOAT)],
            "binary": b"\x00\xff",
            "normal": "string",
        }

        # Encode
        json_str = json.dumps(original, default=json_encoder)

        # Decode
        result = json.loads(json_str, object_hook=json_decoder_hook)

        assert result["arrays"] == [[1, 2, 3], [4, 5]]
        assert result["scalars"] == [EXPECTED_INT, EXPECTED_FLOAT]
        assert result["binary"] == b"\x00\xff"
        assert result["normal"] == "string"
