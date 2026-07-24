"""Tests for KServe datatype → internal DataType mapping."""

import pytest

from trustyai_service.endpoints.consumer import (
    KSERVE_TO_DATATYPE,
    KServeData,
    KServeDataType,
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.endpoints.consumer.consumer_endpoint import process_payload
from trustyai_service.service.data.datasources.data_source import _safe_datatype
from trustyai_service.service.payloads.values.data_type import DataType


class TestKServeToDataTypeMapping:
    """Verify the KSERVE_TO_DATATYPE mapping covers all KServe types."""

    def test_all_kserve_types_mapped(self) -> None:
        """Every KServeDataType has a mapping entry."""
        for kserve_type in KServeDataType:
            assert kserve_type in KSERVE_TO_DATATYPE, (
                f"{kserve_type} missing from KSERVE_TO_DATATYPE"
            )

    @pytest.mark.parametrize(
        ("kserve_type", "expected"),
        [
            (KServeDataType.BOOL, DataType.BOOL),
            (KServeDataType.INT8, DataType.INT32),
            (KServeDataType.INT16, DataType.INT32),
            (KServeDataType.INT32, DataType.INT32),
            (KServeDataType.INT64, DataType.INT64),
            (KServeDataType.UINT8, DataType.INT32),
            (KServeDataType.UINT16, DataType.INT32),
            (KServeDataType.UINT32, DataType.INT64),
            (KServeDataType.UINT64, DataType.INT64),
            (KServeDataType.FP16, DataType.FLOAT),
            (KServeDataType.FP32, DataType.FLOAT),
            (KServeDataType.FP64, DataType.DOUBLE),
            (KServeDataType.BYTES, DataType.STRING),
        ],
    )
    def test_specific_mappings(
        self, kserve_type: KServeDataType, expected: DataType
    ) -> None:
        """Each KServe type maps to the correct internal DataType."""
        assert KSERVE_TO_DATATYPE[kserve_type] == expected


class TestProcessPayloadTypes:
    """Verify process_payload returns correct column types."""

    def test_single_tensor_fp64_returns_double(self) -> None:
        """Single FP64 tensor with shape [3,2] → 2 DOUBLE columns."""
        request = KServeInferenceRequest(
            id="test-1",
            inputs=[
                KServeData(
                    name="features",
                    shape=[3, 2],
                    datatype="FP64",
                    data=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                ),
            ],
        )

        _, names, types = process_payload(request, lambda p: p.inputs)

        assert names == ["features-0", "features-1"]
        assert types == [DataType.DOUBLE, DataType.DOUBLE]

    def test_single_tensor_int32_returns_int32(self) -> None:
        """Single INT32 tensor with shape [2,3] → 3 INT32 columns."""
        request = KServeInferenceRequest(
            id="test-2",
            inputs=[
                KServeData(
                    name="input",
                    shape=[2, 3],
                    datatype="INT32",
                    data=[[1, 2, 3], [4, 5, 6]],
                ),
            ],
        )

        _, names, types = process_payload(request, lambda p: p.inputs)

        assert len(names) == 3  # noqa: PLR2004
        assert types == [DataType.INT32, DataType.INT32, DataType.INT32]

    def test_single_tensor_1d_returns_single_type(self) -> None:
        """Single FP32 tensor with shape [3] → 1 FLOAT column."""
        response = KServeInferenceResponse(
            model_name="test-model",
            id="test-3",
            outputs=[
                KServeData(
                    name="predict",
                    shape=[3],
                    datatype="FP32",
                    data=[0.1, 0.2, 0.3],
                ),
            ],
        )

        _, names, types = process_payload(response, lambda p: p.outputs)

        assert names == ["predict"]
        assert types == [DataType.FLOAT]

    def test_multi_tensor_different_types(self) -> None:
        """Multiple tensors with different types → per-tensor mapping."""
        request = KServeInferenceRequest(
            id="test-4",
            inputs=[
                KServeData(
                    name="age",
                    shape=[2],
                    datatype="FP64",
                    data=[25.0, 30.0],
                ),
                KServeData(
                    name="score",
                    shape=[2],
                    datatype="INT32",
                    data=[100, 200],
                ),
            ],
        )

        _, names, types = process_payload(request, lambda p: p.inputs)

        assert names == ["age", "score"]
        assert types == [DataType.DOUBLE, DataType.INT32]

    def test_bool_type(self) -> None:
        """BOOL tensor maps to DataType.BOOL."""
        request = KServeInferenceRequest(
            id="test-5",
            inputs=[
                KServeData(
                    name="flag",
                    shape=[2],
                    datatype="BOOL",
                    data=[True, False],
                ),
            ],
        )

        _, _, types = process_payload(request, lambda p: p.inputs)

        assert types == [DataType.BOOL]

    def test_bytes_type(self) -> None:
        """BYTES tensor maps to DataType.STRING."""
        request = KServeInferenceRequest(
            id="test-6",
            inputs=[
                KServeData(
                    name="text",
                    shape=[2],
                    datatype="BYTES",
                    data=["hello", "world"],
                ),
            ],
        )

        _, _, types = process_payload(request, lambda p: p.inputs)

        assert types == [DataType.STRING]


class TestSafeDataTypeFallback:
    """Verify _safe_datatype handles missing or corrupted type info."""

    def test_none_types_returns_unknown(self) -> None:
        """None types list falls back to UNKNOWN."""
        assert _safe_datatype(None, 0) == DataType.UNKNOWN

    def test_index_out_of_range_returns_unknown(self) -> None:
        """Index beyond types list falls back to UNKNOWN."""
        assert _safe_datatype(["DOUBLE"], 5) == DataType.UNKNOWN

    def test_invalid_type_string_returns_unknown(self) -> None:
        """Corrupted type string falls back to UNKNOWN."""
        assert _safe_datatype(["INVALID_TYPE"], 0) == DataType.UNKNOWN

    def test_valid_type_string_returns_correct_type(self) -> None:
        """Valid stored type string returns correct DataType."""
        assert _safe_datatype(["DOUBLE", "INT32", "FLOAT"], 0) == DataType.DOUBLE
        assert _safe_datatype(["DOUBLE", "INT32", "FLOAT"], 1) == DataType.INT32
        assert _safe_datatype(["DOUBLE", "INT32", "FLOAT"], 2) == DataType.FLOAT
