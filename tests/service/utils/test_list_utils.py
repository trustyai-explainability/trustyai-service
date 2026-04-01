"""
Tests for list_utils module.

Tests utility functions for working with nested lists and arrays.
Comprehensive row serialization tests are in tests/service/serialization/test_rows.py.
"""

import numpy as np

from src.service.data.storage.pvc import MAX_VOID_TYPE_LENGTH
from src.service.serialization import deserialize_rows, serialize_rows
from src.service.utils.list_utils import contains_non_numeric, get_list_shape


class TestGetListShape:
    """Test get_list_shape function."""

    def test_empty_list(self):
        """Test shape of empty list."""
        assert get_list_shape([]) == []

    def test_1d_list(self):
        """Test shape of 1D list."""
        lst = [1, 2, 3, 4, 5]
        assert get_list_shape(lst) == [5]

    def test_2d_list(self):
        """Test shape of 2D list."""
        lst = [[1, 2], [3, 4], [5, 6]]
        assert get_list_shape(lst) == [3, 2]

    def test_3d_list(self):
        """Test shape of 3D list."""
        lst = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        assert get_list_shape(lst) == [2, 2, 2]

    def test_nested_list(self):
        """Test shape of nested list."""
        # Note: get_list_shape assumes non-jagged, non-empty lists
        lst = [[1], [2], [3]]
        assert get_list_shape(lst) == [3, 1]

    def test_deeply_nested_list(self):
        """Test shape of deeply nested list."""
        lst = [[[[1]]]]
        assert get_list_shape(lst) == [1, 1, 1, 1]


class TestContainsNonNumeric:
    """Test contains_non_numeric function."""

    def test_numeric_list(self):
        """Test list with only numeric values."""
        lst = [1, 2, 3, 4.5, 6.7]
        assert not contains_non_numeric(lst)

    def test_string_in_list(self):
        """Test list containing strings."""
        lst = [1, 2, "string", 4]
        assert contains_non_numeric(lst)

    def test_bool_in_list(self):
        """Test list containing booleans."""
        lst = [1, 2, True, 4]
        assert contains_non_numeric(lst)

    def test_nested_numeric_list(self):
        """Test nested list with only numeric values."""
        lst = [[1, 2], [3, 4], [5.5, 6.7]]
        assert not contains_non_numeric(lst)

    def test_nested_non_numeric_list(self):
        """Test nested list with non-numeric values."""
        lst = [[1, 2], ["text", 4], [5, 6]]
        assert contains_non_numeric(lst)

    def test_numpy_array_numeric(self):
        """Test numpy array with numeric values."""
        arr = np.array([1, 2, 3, 4])
        assert not contains_non_numeric(arr)

    def test_numpy_array_strings(self):
        """Test numpy array with strings."""
        arr = np.array(["a", "b", "c"])
        assert contains_non_numeric(arr)

    def test_deeply_nested_with_string(self):
        """Test deeply nested list with string."""
        lst = [[[1, 2]], [[3, "text"]]]
        assert contains_non_numeric(lst)


class TestSerializationReExports:
    """Test that serialization functions are re-exported correctly from serialization package."""

    def test_serialize_rows_is_callable(self):
        """Test serialize_rows is available."""
        assert callable(serialize_rows)

    def test_deserialize_rows_is_callable(self):
        """Test deserialize_rows is available."""
        assert callable(deserialize_rows)

    def test_basic_roundtrip(self):
        """Test basic serialization roundtrip works through re-export."""
        rows = [[1, 2, 3], [4, 5, 6]]
        serialized = serialize_rows(rows, max_void_type_length=MAX_VOID_TYPE_LENGTH)
        deserialized = deserialize_rows(serialized)

        assert len(deserialized) == 2
        assert list(deserialized[0]) == [1, 2, 3]
        assert list(deserialized[1]) == [4, 5, 6]
