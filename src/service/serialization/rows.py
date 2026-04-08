"""
Row-level serialization for inference data.

Serializes nested lists/dicts into numpy void arrays with dynamic sizing
using JSON with gzip compression for security and compatibility.
"""

import gzip
import json
import logging

import numpy as np

from .detection import is_gzip
from .encoders import json_encoder

logger = logging.getLogger(__name__)


def serialize_rows(lst: list | np.ndarray, max_void_type_length: int) -> np.ndarray:
    """
    Convert a nested list to a 1D numpy array with dynamic void type sizing.

    Uses JSON with gzip compression for security and compatibility. Each element
    contains a bytes serialization of the corresponding row. The void type size
    is computed to fit the largest serialized row, preventing silent truncation
    while optimizing storage.

    Args:
        lst: List of rows to serialize
        max_void_type_length: Maximum allowed void type size (raises error if exceeded)

    Returns:
        np.ndarray with dtype V{size} where size is the maximum serialized row size

    Raises:
        ValueError: If any serialized row exceeds max_void_type_length

    Examples:
        >>> rows = [[1, 2, 3], [4, 5, 6]]
        >>> serialized = serialize_rows(rows, max_void_type_length=1024)
        >>> serialized.dtype.kind
        'V'
    """
    # Serialize all rows first to compute required size (using JSON + gzip)
    serialized = []
    for row in lst:
        json_str = json.dumps(row, default=json_encoder)
        compressed = gzip.compress(json_str.encode("utf-8"))
        serialized.append(compressed)

    # Compute required void type size (maximum of all serialized rows)
    max_size = max(len(s) for s in serialized) if serialized else 0

    # Validate against maximum allowed size
    if max_size > max_void_type_length:
        raise ValueError(
            f"Serialized row size {max_size} bytes exceeds maximum allowed size "
            f"{max_void_type_length} bytes. Consider reducing payload size, using compression, "
            f"or increasing MAX_VOID_TYPE_LENGTH configuration."
        )

    # Use dynamic void type based on actual data size (prevents truncation and saves space)
    void_dtype = f"V{max_size}" if max_size > 0 else f"V{max_void_type_length}"
    return np.array([np.void(s) for s in serialized], dtype=void_dtype)


def deserialize_rows(serialized: np.ndarray) -> np.ndarray:
    """
    Convert a 1D numpy array from `serialize_rows` to a numpy object array.

    Deserializes gzip-compressed JSON data.

    Args:
        serialized: Numpy array of serialized rows (from serialize_rows)

    Returns:
        Numpy object array containing deserialized rows

    Raises:
        ValueError: If deserialization fails for any row

    Examples:
        >>> rows = [[1, 2, 3], [4, 5, 6]]
        >>> serialized = serialize_rows(rows, max_void_type_length=1024)
        >>> deserialized = deserialize_rows(serialized)
        >>> list(deserialized[0])
        [1, 2, 3]
    """
    deserialized = []

    for row in serialized:
        # Convert numpy void to bytes
        row_bytes = bytes(row)

        if is_gzip(row_bytes):
            # Gzip-compressed JSON (production format)
            # Gzip decompressor handles the data correctly without manual null-byte stripping
            try:
                json_str = gzip.decompress(row_bytes).decode("utf-8")
                deserialized.append(json.loads(json_str))
            except (OSError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(f"Failed to deserialize row as gzip-compressed JSON: {e}") from e
        else:
            raise ValueError(
                f"Unsupported serialization format. Expected gzip-compressed JSON. First bytes: {row_bytes[:10]!r}"
            )

    return np.array(deserialized, dtype="O")
