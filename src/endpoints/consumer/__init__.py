from typing import Optional, Dict, List, Literal

from pydantic import BaseModel


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


class KServeData(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    parameters: Optional[Dict[str, str]] = None
    data: List


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
