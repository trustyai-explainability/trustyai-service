import pickle
from typing import Any

import numpy as np


def get_list_shape(lst: list) -> list[int]:
    """Get the shape of a nested list, assuming the sublists are not jagged"""
    return [len(lst)] + get_list_shape(lst[0]) if isinstance(lst, list) else []


def contains_non_numeric(lst: list | np.ndarray | Any) -> bool:
    """Check if an arbitrarily deep nested list contains any non-numeric elements"""
    if isinstance(lst, (list, np.ndarray)):
        return any(contains_non_numeric(item) for item in lst)
    else:
        return isinstance(lst, (bool, str))


def serialize_rows(lst: list | np.ndarray, max_void_type_length: int) -> np.ndarray:
    """
    Convert a nested list to a 1D numpy array with dynamic void type sizing.

    Each element contains a bytes serialization of the corresponding row.
    The void type size is computed to fit the largest serialized row,
    preventing silent truncation while optimizing storage.

    Args:
        lst: List of rows to serialize
        max_void_type_length: Maximum allowed void type size (raises error if exceeded)

    Returns:
        np.ndarray with dtype V{size} where size is the maximum serialized row size

    Raises:
        ValueError: If any serialized row exceeds max_void_type_length
    """
    # Serialize all rows first to compute required size
    serialized = [pickle.dumps(row) for row in lst]

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
    """Convert a 1D numpy array from `serialize_rows` to a numpy object array"""
    deserialized = [pickle.loads(row) for row in serialized]
    return np.array(deserialized, dtype="O")
