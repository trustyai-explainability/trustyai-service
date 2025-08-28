from typing import Optional, Dict, List, Literal, Any
from enum import Enum
from pydantic import BaseModel, model_validator, ConfigDict
import numpy as np


PartialKind = Literal["request", "response"]

class PartialPayloadId(BaseModel):
    prediction_id: Optional[str] = None
    kind: Optional[PartialKind] = None

    def get_prediction_id(self) -> str:
        return self.prediction_id

    def set_prediction_id(self, id: str):
        self.prediction_id = id

    def get_kind(self) -> PartialKind:
        return self.kind

    def set_kind(self, kind: PartialKind):
        self.kind = kind


class InferencePartialPayload(BaseModel):
    partialPayloadId: Optional[PartialPayloadId] = None
    metadata: Optional[Dict[str, str]] = {}
    data: Optional[str] = None
    modelid: Optional[str] = None

    def get_id(self) -> str:
        return self.partialPayloadId.prediction_id if self.partialPayloadId else None

    def set_id(self, id: str):
        if not self.partialPayloadId:
            self.partialPayloadId = PartialPayloadId()
        self.partialPayloadId.prediction_id = id

    def get_kind(self) -> PartialKind:
        return self.partialPayloadId.kind if self.partialPayloadId else None

    def set_kind(self, kind: PartialKind):
        if not self.partialPayloadId:
            self.partialPayloadId = PartialPayloadId()
        self.partialPayloadId.kind = kind

    def get_model_id(self) -> str:
        return self.modelid

    def set_model_id(self, model_id: str):
        self.modelid = model_id


class KServeDataType(str, Enum):
    BOOL  = "BOOL"
    INT8  = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    UINT8  = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    FP16 = "FP16"
    FP32 = "FP32"
    FP64 = "FP64"
    BYTES = "BYTES" 

K_SERVE_NUMPY_DTYPES = {
    KServeDataType.INT8:  np.int8,
    KServeDataType.INT16: np.int16,
    KServeDataType.INT32: np.int32,
    KServeDataType.INT64: np.int64,
    KServeDataType.UINT8:  np.uint8,
    KServeDataType.UINT16: np.uint16,
    KServeDataType.UINT32: np.uint32,
    KServeDataType.UINT64: np.uint64,
    KServeDataType.FP16: np.float16,
    KServeDataType.FP32: np.float32,
    KServeDataType.FP64: np.float64,
}

class KServeData(BaseModel):
    
    model_config = ConfigDict(use_enum_values=True)

    name: str
    shape: List[int]
    datatype: KServeDataType
    parameters: Optional[Dict[str, str]] = None
    data: List[Any]

    @model_validator(mode="after")
    def _validate_shape(self) -> "KServeData":
        raw = np.array(self.data, dtype=object)
        actual = tuple(raw.shape)
        declared = tuple(self.shape)
        if declared != actual:
            raise ValueError(
                f"Declared shape {declared} does not match data shape {actual}"
            )
        return self

    @model_validator(mode="after")
    def validate_data_matches_type(self) -> "KServeData":
        flat = np.array(self.data, dtype=object).flatten()

        if self.datatype == KServeDataType.BYTES:
            for v in flat:
                if not isinstance(v, str):
                    raise ValueError(
                        f"All values must be JSON strings for datatype {self.datatype}; "
                        f"found {type(v).__name__}: {v}"
                    )
            return self

        if self.datatype == KServeDataType.BOOL:
            for v in flat:
                if not (isinstance(v, (bool, int)) and v in (0, 1, True, False)):
                    raise ValueError(
                        f"All values must be bool or 0/1 for datatype {self.datatype}; found {v}"
                    )
            return self

        np_dtype = K_SERVE_NUMPY_DTYPES.get(self.datatype)
        if np_dtype is None:
            raise ValueError(f"Unsupported datatype: {self.datatype}")

        if np.dtype(np_dtype).kind == "u":
            for v in flat:
                if isinstance(v, (int, float)) and v < 0:
                    raise ValueError(
                        f"Negative value {v} not allowed for unsigned type {self.datatype}"
                    )

        try:
            np.array(flat, dtype=np_dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Data cannot be cast to {self.datatype}: {e}")

        return self

class KServeInferenceRequest(BaseModel):
    id: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None
    inputs: List[KServeData]
    outputs: Optional[List[KServeData]] = None


class KServeInferenceResponse(BaseModel):
    model_name: str = None
    model_version: Optional[str] = None
    id: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None
    outputs: List[KServeData]
