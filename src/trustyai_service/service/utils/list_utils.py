"""List utility functions.

Provides utilities for working with nested lists and arrays.
"""

from __future__ import annotations

import numbers

import numpy as np


def get_list_shape(lst: list) -> list[int]:
    """Get the shape of a nested list, assuming the sublists are not jagged.

    Args:
        lst: Nested list to analyze

    Returns:
        List of dimensions (shape)

    Examples:
        >>> get_list_shape([1, 2, 3])
        [3]
        >>> get_list_shape([[1, 2], [3, 4]])
        [2, 2]
        >>> get_list_shape([[]])
        [1, 0]

    """
    if not isinstance(lst, list) or not lst:
        return []
    # Handle empty inner lists
    if isinstance(lst[0], list) and not lst[0]:
        return [len(lst), 0]
    return [len(lst), *get_list_shape(lst[0])]


def contains_non_numeric(lst: object) -> bool:
    """Check if an arbitrarily deep nested list contains any non-numeric elements.

    Args:
        lst: List or array to check

    Returns:
        True if any element is not a numeric type (including dict, bytes, None, bool, str, custom objects)

    Examples:
        >>> contains_non_numeric([1, 2, 3])
        False
        >>> contains_non_numeric([1, "two", 3])
        True
        >>> contains_non_numeric([[1, 2], [True, 4]])
        True
        >>> contains_non_numeric([{"key": "value"}])
        True
        >>> contains_non_numeric([b"bytes"])
        True

    """
    if isinstance(lst, (list, np.ndarray)):
        return any(contains_non_numeric(item) for item in lst)
    # Bools are technically numbers in Python (bool is subclass of int),
    # but we treat them as non-numeric for serialization type preservation
    if isinstance(lst, bool):
        return True
    # Consider a scalar non-numeric if it's not a number
    # Use numbers.Number to catch Python numerics and np.number for NumPy scalars
    return not isinstance(lst, (numbers.Number, np.number))
