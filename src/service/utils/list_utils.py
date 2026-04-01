"""
List utility functions.

Provides utilities for working with nested lists and arrays.
"""

from typing import Any

import numpy as np


def get_list_shape(lst: list) -> list[int]:
    """
    Get the shape of a nested list, assuming the sublists are not jagged.

    Args:
        lst: Nested list to analyze

    Returns:
        List of dimensions (shape)

    Examples:
        >>> get_list_shape([1, 2, 3])
        [3]
        >>> get_list_shape([[1, 2], [3, 4]])
        [2, 2]
    """
    return [len(lst)] + get_list_shape(lst[0]) if isinstance(lst, list) and lst else []


def contains_non_numeric(lst: list | np.ndarray | Any) -> bool:
    """
    Check if an arbitrarily deep nested list contains any non-numeric elements.

    Args:
        lst: List or array to check

    Returns:
        True if any element is a bool or string, False otherwise

    Examples:
        >>> contains_non_numeric([1, 2, 3])
        False
        >>> contains_non_numeric([1, "two", 3])
        True
        >>> contains_non_numeric([[1, 2], [True, 4]])
        True
    """
    if isinstance(lst, (list, np.ndarray)):
        return any(contains_non_numeric(item) for item in lst)
    else:
        return isinstance(lst, (bool, str))
