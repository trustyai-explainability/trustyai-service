"""Tests for KServeData shape and datatype validation."""

import unittest

import pytest

from trustyai_service.endpoints.consumer import KServeData


class TestKServeDataShapeValidation(unittest.TestCase):
    """Test that _validate_shape accepts flat tensors and reshapes them."""

    def test_flat_data_reshaped_to_column_vector(self) -> None:
        """Flat list with shape [5, 1] is reshaped to column vector."""
        tensor = KServeData(
            name="output",
            shape=[5, 1],
            datatype="FP32",
            data=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        assert tensor.data == [[0.1], [0.2], [0.3], [0.4], [0.5]]

    def test_flat_data_reshaped_to_matrix(self) -> None:
        """Flat list with shape [2, 3] is reshaped to 2x3 matrix."""
        tensor = KServeData(
            name="output",
            shape=[2, 3],
            datatype="INT32",
            data=[1, 2, 3, 4, 5, 6],
        )
        assert tensor.data == [[1, 2, 3], [4, 5, 6]]

    def test_already_shaped_data_unchanged(self) -> None:
        """Pre-shaped data matching declared shape passes unchanged."""
        tensor = KServeData(
            name="output",
            shape=[2, 3],
            datatype="INT32",
            data=[[1, 2, 3], [4, 5, 6]],
        )
        assert tensor.data == [[1, 2, 3], [4, 5, 6]]

    def test_scalar_shape(self) -> None:
        """Single-element tensor with shape [1] passes."""
        tensor = KServeData(
            name="output",
            shape=[1],
            datatype="FP32",
            data=[42.0],
        )
        assert tensor.data == [42.0]

    def test_1d_vector(self) -> None:
        """1D vector with shape [5] passes without reshape."""
        tensor = KServeData(
            name="output",
            shape=[5],
            datatype="FP32",
            data=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        assert tensor.data == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_wrong_element_count_raises(self) -> None:
        """Mismatched element count raises ValueError."""
        with pytest.raises(ValueError, match=r"requires 5 elements but got 3"):
            KServeData(
                name="output",
                shape=[5, 1],
                datatype="FP32",
                data=[1.0, 2.0, 3.0],
            )

    def test_3d_flat_data_reshaped(self) -> None:
        """Flat list reshaped to 3D tensor."""
        tensor = KServeData(
            name="output",
            shape=[2, 2, 2],
            datatype="INT32",
            data=[1, 2, 3, 4, 5, 6, 7, 8],
        )
        assert tensor.data == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

    def test_empty_tensor(self) -> None:
        """Zero-element tensor with shape [0] and empty data passes."""
        tensor = KServeData(
            name="output",
            shape=[0],
            datatype="FP32",
            data=[],
        )
        assert tensor.data == []

    def test_empty_shape_with_data_raises(self) -> None:
        """Zero-element shape with non-empty data raises ValueError."""
        with pytest.raises(ValueError, match=r"requires 0 elements but got 1"):
            KServeData(
                name="output",
                shape=[0],
                datatype="FP32",
                data=[1.0],
            )

    def test_flat_reshape_then_type_validation_unsigned(self) -> None:
        """Flat data is reshaped before type validation catches negative unsigned values."""
        with pytest.raises(ValueError, match=r"Negative value"):
            KServeData(
                name="output",
                shape=[2, 2],
                datatype="UINT32",
                data=[1, 2, -3, 4],
            )

    def test_flat_reshape_then_type_validation_passes(self) -> None:
        """Flat data is reshaped and then passes type validation end-to-end."""
        tensor = KServeData(
            name="output",
            shape=[2, 2],
            datatype="UINT32",
            data=[1, 2, 3, 4],
        )
        assert tensor.data == [[1, 2], [3, 4]]


if __name__ == "__main__":
    unittest.main()
