"""Tests for process_payload and reconciliation error helpers.

Covers single-tensor, multi-tensor, shape-mismatch, row-count-mismatch,
non-numeric data, and the error-raising helper functions.
"""

import unittest

import numpy as np
import pytest

from src.endpoints.consumer import KServeData, KServeInferenceRequest
from src.endpoints.consumer.consumer_endpoint import (
    process_payload,
    reconcile_mismatching_row_count_error,
    reconcile_mismatching_shape_error,
)
from src.exceptions import ReconciliationError


class TestProcessPayloadSingleTensor(unittest.TestCase):
    """process_payload with a single input/output tensor."""

    def test_1d_single_tensor(self) -> None:
        """Single tensor with shape [N] produces N rows, 1 column."""
        payload = KServeInferenceRequest(
            id="t-1d",
            inputs=[
                KServeData(
                    name="feature",
                    shape=[4],
                    datatype="FP32",
                    data=[1.0, 2.0, 3.0, 4.0],
                ),
            ],
        )
        arr, names = process_payload(payload, lambda p: p.inputs)
        assert names == ["feature"]
        assert arr.shape == (4,)
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0, 4.0])

    def test_2d_single_tensor_column_names(self) -> None:
        """Single tensor with shape [N, D] generates D column names."""
        payload = KServeInferenceRequest(
            id="t-2d",
            inputs=[
                KServeData(
                    name="feat",
                    shape=[3, 2],
                    datatype="FP32",
                    data=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                ),
            ],
        )
        arr, names = process_payload(payload, lambda p: p.inputs)
        assert names == ["feat-0", "feat-1"]
        assert arr.shape == (3, 2)

    def test_single_tensor_enforced_shape_match(self) -> None:
        """When enforced_first_shape matches, no error."""
        expected_rows = 5
        payload = KServeInferenceRequest(
            id="t-enforce-ok",
            inputs=[
                KServeData(
                    name="x",
                    shape=[expected_rows],
                    datatype="INT32",
                    data=list(range(expected_rows)),
                ),
            ],
        )
        arr, names = process_payload(
            payload, lambda p: p.inputs, enforced_first_shape=expected_rows
        )
        assert len(arr) == expected_rows
        assert names == ["x"]

    def test_single_tensor_enforced_shape_mismatch(self) -> None:
        """Mismatched enforced_first_shape raises ReconciliationError."""
        payload = KServeInferenceRequest(
            id="t-enforce-bad",
            inputs=[
                KServeData(
                    name="x",
                    shape=[3],
                    datatype="INT32",
                    data=[0, 1, 2],
                ),
            ],
        )
        with pytest.raises(ReconciliationError, match="number of"):
            process_payload(payload, lambda p: p.inputs, enforced_first_shape=5)


class TestProcessPayloadMultiTensor(unittest.TestCase):
    """process_payload with multiple input/output tensors."""

    def test_multi_tensor_matching_shapes(self) -> None:
        """Multiple tensors with the same shape are concatenated column-wise."""
        payload = KServeInferenceRequest(
            id="t-multi",
            inputs=[
                KServeData(
                    name="age",
                    shape=[3],
                    datatype="FP32",
                    data=[25.0, 30.0, 35.0],
                ),
                KServeData(
                    name="income",
                    shape=[3],
                    datatype="FP32",
                    data=[50000.0, 60000.0, 70000.0],
                ),
            ],
        )
        arr, names = process_payload(payload, lambda p: p.inputs)
        assert names == ["age", "income"]
        assert arr.shape == (3, 2)
        np.testing.assert_array_almost_equal(arr[:, 0], [25.0, 30.0, 35.0])
        np.testing.assert_array_almost_equal(arr[:, 1], [50000.0, 60000.0, 70000.0])

    def test_multi_tensor_mismatched_shapes_raises(self) -> None:
        """Multiple tensors with different shapes raise ReconciliationError."""
        payload = KServeInferenceRequest(
            id="t-mismatch",
            inputs=[
                KServeData(
                    name="a",
                    shape=[3],
                    datatype="FP32",
                    data=[1.0, 2.0, 3.0],
                ),
                KServeData(
                    name="b",
                    shape=[2],
                    datatype="FP32",
                    data=[4.0, 5.0],
                ),
            ],
        )
        with pytest.raises(ReconciliationError, match="shapes were mismatched"):
            process_payload(payload, lambda p: p.inputs)

    def test_multi_tensor_enforced_shape_mismatch(self) -> None:
        """Multi-tensor: enforced row count differs from actual."""
        payload = KServeInferenceRequest(
            id="t-multi-enforce",
            inputs=[
                KServeData(
                    name="a",
                    shape=[3],
                    datatype="FP32",
                    data=[1.0, 2.0, 3.0],
                ),
                KServeData(
                    name="b",
                    shape=[3],
                    datatype="FP32",
                    data=[4.0, 5.0, 6.0],
                ),
            ],
        )
        with pytest.raises(ReconciliationError, match="number of"):
            process_payload(payload, lambda p: p.inputs, enforced_first_shape=5)


class TestProcessPayloadNonNumeric(unittest.TestCase):
    """process_payload with non-numeric (string/BYTES) data."""

    def test_single_tensor_non_numeric(self) -> None:
        """Single tensor with BYTES data uses object dtype."""
        payload = KServeInferenceRequest(
            id="t-bytes",
            inputs=[
                KServeData(
                    name="category",
                    shape=[3],
                    datatype="BYTES",
                    data=["cat", "dog", "bird"],
                ),
            ],
        )
        arr, names = process_payload(payload, lambda p: p.inputs)
        assert names == ["category"]
        assert arr.dtype == object
        assert list(arr) == ["cat", "dog", "bird"]

    def test_multi_tensor_non_numeric(self) -> None:
        """Multiple tensors with BYTES data use object dtype."""
        payload = KServeInferenceRequest(
            id="t-bytes-multi",
            inputs=[
                KServeData(
                    name="color",
                    shape=[2],
                    datatype="BYTES",
                    data=["red", "blue"],
                ),
                KServeData(
                    name="size",
                    shape=[2],
                    datatype="BYTES",
                    data=["large", "small"],
                ),
            ],
        )
        arr, names = process_payload(payload, lambda p: p.inputs)
        assert names == ["color", "size"]
        assert arr.dtype == object
        assert arr.shape == (2, 2)


class TestReconciliationErrorHelpers(unittest.TestCase):
    """Test the error-raising helper functions directly."""

    def test_mismatching_shape_error_includes_details(self) -> None:
        """reconcile_mismatching_shape_error message includes all tensors."""
        shape_tuples = [("feat_a", [3, 2]), ("feat_b", [4, 2])]
        with pytest.raises(ReconciliationError, match="feat_a") as exc_info:
            reconcile_mismatching_shape_error(shape_tuples, "input", "payload-xyz")
        assert "feat_b" in str(exc_info.value)
        assert "input shapes were mismatched" in str(exc_info.value)
        assert exc_info.value.payload_id == "payload-xyz"

    def test_mismatching_row_count_error_includes_counts(self) -> None:
        """reconcile_mismatching_row_count_error includes both counts."""
        with pytest.raises(ReconciliationError) as exc_info:
            reconcile_mismatching_row_count_error("payload-abc", 10, 7)
        err_msg = str(exc_info.value)
        assert "10" in err_msg
        assert "7" in err_msg
        assert exc_info.value.payload_id == "payload-abc"

    def test_mismatching_shape_error_output_type(self) -> None:
        """Error message uses 'output' when enforced_first_shape is set."""
        shape_tuples = [("out_a", [5]), ("out_b", [3])]
        with pytest.raises(ReconciliationError, match="output shapes"):
            reconcile_mismatching_shape_error(shape_tuples, "output", "payload-out")


if __name__ == "__main__":
    unittest.main()
