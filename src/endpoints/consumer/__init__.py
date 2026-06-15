"""Inference payload consumer endpoints."""

from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

PartialKind = Literal["request", "response"]


class InferencePartialPayload(BaseModel):
    """Partial inference payload for KServe agent uploads.

    Flat structure matching the Java TrustyAI service wire format.
    """

    id: str | None = None
    kind: PartialKind | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    data: str | None = None
    modelid: str | None = None


class KServeDataType(StrEnum):
    """KServe inference protocol data types."""

    BOOL = "BOOL"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    FP16 = "FP16"
    FP32 = "FP32"
    FP64 = "FP64"
    BYTES = "BYTES"


K_SERVE_NUMPY_DTYPES = {
    KServeDataType.INT8: np.int8,
    KServeDataType.INT16: np.int16,
    KServeDataType.INT32: np.int32,
    KServeDataType.INT64: np.int64,
    KServeDataType.UINT8: np.uint8,
    KServeDataType.UINT16: np.uint16,
    KServeDataType.UINT32: np.uint32,
    KServeDataType.UINT64: np.uint64,
    KServeDataType.FP16: np.float16,
    KServeDataType.FP32: np.float32,
    KServeDataType.FP64: np.float64,
}


class InferParameter(BaseModel):
    """KServe V2 inference parameter supporting bool, int, or string values."""

    bool_param: bool | None = None
    int_param: int | None = None
    string_param: str | None = None


class KServeData(BaseModel):
    """KServe tensor data with shape, type, and validation."""

    model_config = ConfigDict(use_enum_values=True)

    name: str
    shape: list[int]
    datatype: KServeDataType
    # KServe V2 spec: parameters should be dict[str, InferParameter]
    # Also accept dict[str, str] for backward compatibility
    parameters: dict[str, InferParameter | str] | None = None
    data: list[Any]

    @model_validator(mode="after")
    def _validate_shape(self) -> "KServeData":
        raw = np.array(self.data, dtype=object)
        declared = tuple(self.shape)
        expected_elements = int(np.prod(declared))
        actual_elements = raw.size
        if expected_elements != actual_elements:
            msg = f"Declared shape {declared} requires {expected_elements} elements but got {actual_elements}"
            raise ValueError(msg)
        self.data = raw.reshape(declared).tolist()
        return self

    @model_validator(mode="after")
    def validate_data_matches_type(self) -> "KServeData":
        """Validate data values match declared datatype."""
        flat = np.array(self.data, dtype=object).flatten()

        if self.datatype == KServeDataType.BYTES:
            for v in flat:
                if not isinstance(v, str):
                    msg = f"All values must be JSON strings for datatype {self.datatype}; found {type(v).__name__}: {v}"
                    raise TypeError(
                        msg,
                    )
            return self

        if self.datatype == KServeDataType.BOOL:
            for v in flat:
                if not (isinstance(v, (bool, int)) and v in (0, 1, True, False)):
                    msg = f"All values must be bool or 0/1 for datatype {self.datatype}; found {v}"
                    raise ValueError(
                        msg,
                    )
            return self

        np_dtype = K_SERVE_NUMPY_DTYPES.get(self.datatype)
        if np_dtype is None:
            msg = f"Unsupported datatype: {self.datatype}"
            raise ValueError(msg)

        if np.dtype(np_dtype).kind == "u":
            for v in flat:
                if isinstance(v, (int, float)) and v < 0:
                    msg = f"Negative value {v} not allowed for unsigned type {self.datatype}"
                    raise ValueError(
                        msg,
                    )

        try:
            np.array(flat, dtype=np_dtype)
        except (ValueError, TypeError) as e:
            msg = f"Data cannot be cast to {self.datatype}: {e}"
            raise ValueError(msg) from e

        return self


class KServeInferenceRequest(BaseModel):
    """KServe V2 inference request."""

    id: str | None = None
    parameters: dict[str, str] | None = None
    inputs: list[KServeData]
    outputs: list[KServeData] | None = None


class KServeInferenceResponse(BaseModel):
    """KServe V2 inference response."""

    model_name: str | None = None
    model_version: str | None = None
    id: str | None = None
    parameters: dict[str, str] | None = None
    outputs: list[KServeData]
