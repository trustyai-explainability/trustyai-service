"""Tests for KServe datatype → internal DataType mapping."""

from pathlib import Path
from unittest.mock import patch

import pytest

from trustyai_service.endpoints.consumer import (
    KSERVE_TO_DATATYPE,
    KServeData,
    KServeDataType,
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.endpoints.consumer.consumer_endpoint import (
    process_payload,
)
from trustyai_service.service.data.datasources.data_source import (
    DataSource,
    _safe_datatype,
)
from trustyai_service.service.data.storage.pvc import PVCStorage
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


_STORAGE_PATCH = "trustyai_service.service.data.model_data.get_global_storage_interface"


class TestDataTypeRoundTrip:
    """Verify data types survive the full round-trip through PVC storage and get_metadata().

    Tests the chain: set_column_types() → HDF5 attribute → get_column_types() → get_metadata() schema.
    """

    @pytest.fixture
    def pvc_storage(self, tmp_path: Path) -> PVCStorage:
        """Create a temporary PVC storage backend."""
        return PVCStorage(str(tmp_path), data_file="test.hdf5")

    @pytest.mark.asyncio
    async def test_fp64_roundtrip(self, pvc_storage: PVCStorage) -> None:
        """FP64 inputs + FP32 output persist as DOUBLE/FLOAT in get_metadata() schema."""
        await pvc_storage.write_data(
            "m_inputs", [[25.0, 50000.0], [30.0, 60000.0]], ["age", "income"]
        )
        await pvc_storage.write_data("m_outputs", [[0.95], [0.42]], ["score"])
        await pvc_storage.write_data("m_metadata", [["t"], ["t"]], ["tags"])
        await pvc_storage.set_column_types("m_inputs", ["DOUBLE", "DOUBLE"])
        await pvc_storage.set_column_types("m_outputs", ["FLOAT"])

        with patch(_STORAGE_PATCH, return_value=pvc_storage):
            data_source = DataSource()
            data_source.storage_interface = pvc_storage
            metadata = await data_source.get_metadata("m")

        assert metadata.input_schema.items["age"].type == DataType.DOUBLE
        assert metadata.input_schema.items["income"].type == DataType.DOUBLE
        assert metadata.output_schema.items["score"].type == DataType.FLOAT

    @pytest.mark.asyncio
    async def test_mixed_types_roundtrip(self, pvc_storage: PVCStorage) -> None:
        """INT32 + BOOL inputs persist with correct types, not STRING."""
        await pvc_storage.write_data(
            "mx_inputs", [[10, True], [20, False]], ["count", "flag"]
        )
        await pvc_storage.write_data("mx_outputs", [[1], [0]], ["pred"])
        await pvc_storage.write_data("mx_metadata", [["t"], ["t"]], ["tags"])
        await pvc_storage.set_column_types("mx_inputs", ["INT32", "BOOL"])
        await pvc_storage.set_column_types("mx_outputs", ["INT64"])

        with patch(_STORAGE_PATCH, return_value=pvc_storage):
            data_source = DataSource()
            data_source.storage_interface = pvc_storage
            metadata = await data_source.get_metadata("mx")

        assert metadata.input_schema.items["count"].type == DataType.INT32
        assert metadata.input_schema.items["flag"].type == DataType.BOOL
        assert metadata.output_schema.items["pred"].type == DataType.INT64

    @pytest.mark.asyncio
    async def test_old_data_without_types_returns_unknown(
        self, pvc_storage: PVCStorage
    ) -> None:
        """Data written without set_column_types falls back to UNKNOWN, not STRING."""
        await pvc_storage.write_data("legacy_inputs", [[1.0, 2.0]], ["f1", "f2"])
        await pvc_storage.write_data("legacy_outputs", [[0.5]], ["out"])
        await pvc_storage.write_data("legacy_metadata", [["tag"]], ["tags"])

        with patch(_STORAGE_PATCH, return_value=pvc_storage):
            data_source = DataSource()
            data_source.storage_interface = pvc_storage
            metadata = await data_source.get_metadata("legacy")

        assert metadata.input_schema.items["f1"].type == DataType.UNKNOWN
        assert metadata.output_schema.items["out"].type == DataType.UNKNOWN

    def test_process_payload_types_match_storage_values(self) -> None:
        """Types from process_payload() are valid DataType.value strings for set_column_types()."""
        request = KServeInferenceRequest(
            id="chain-test",
            inputs=[
                KServeData(name="x", shape=[1], datatype="FP64", data=[1.0]),
                KServeData(name="y", shape=[1], datatype="INT32", data=[42]),
            ],
        )
        _, _, types = process_payload(request, lambda p: p.inputs)

        for dt in types:
            assert isinstance(dt, DataType)
            assert dt.value == DataType(dt.value).value
