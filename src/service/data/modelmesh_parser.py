"""
ModelMesh protobuf payload parsing.
"""

import base64
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

try:
    from src.proto.grpc_predict_v2_pb2 import (
        ModelInferRequest,
        ModelInferResponse,
        InferTensorContents,
    )
except ImportError as e:
    raise ImportError("Protobuf modules not found. Make sure to run the generate_protos.py script first.") from e


class PartialPayload(BaseModel):
    """
    A partial payload (either input or output) from ModelMesh.
    """

    data: str
    metadata: Dict[str, str] = {}


class ModelMeshPayloadParser:
    """
    ModelMesh protobuf payloads.
    """

    MM_MODEL_SUFFIX = "__isvc"
    BIAS_IGNORE_PARAM = "bias_ignore"

    @staticmethod
    def standardize_model_id(model_id: str) -> str:
        """Standardize the model ID by removing the ModelMesh suffix if present."""
        if model_id and ModelMeshPayloadParser.MM_MODEL_SUFFIX in model_id:
            index = model_id.rfind(ModelMeshPayloadParser.MM_MODEL_SUFFIX)
            return model_id[:index]
        return model_id

    @staticmethod
    def parse_input_payload(payload: PartialPayload) -> ModelInferRequest:
        """Parse the input payload from base64 to ModelInferRequest."""
        try:
            input_bytes = base64.b64decode(payload.data)
            input_request = ModelInferRequest()
            input_request.ParseFromString(input_bytes)
            return input_request
        except Exception as e:
            raise ValueError(f"Failed to parse input payload: {e}") from e

    @staticmethod
    def parse_output_payload(payload: PartialPayload) -> ModelInferResponse:
        """Parse the output payload from base64 to ModelInferResponse."""
        try:
            output_bytes = base64.b64decode(payload.data)
            output_response = ModelInferResponse()
            output_response.ParseFromString(output_bytes)
            return output_response
        except Exception as e:
            raise ValueError(f"Failed to parse output payload: {e}") from e

    @staticmethod
    def _extract_tensor_data(tensor, tensor_type: str) -> np.ndarray:
        """Extract data from a tensor based on its type."""
        if tensor_type == "BOOL":
            return np.array(tensor.contents.bool_contents)
        elif tensor_type in {"INT8", "INT16", "INT32"}:
            return np.array(tensor.contents.int_contents)
        elif tensor_type == "INT64":
            return np.array(tensor.contents.int64_contents)
        elif tensor_type in {"UINT8", "UINT16", "UINT32"}:
            return np.array(tensor.contents.uint_contents)
        elif tensor_type == "UINT64":
            return np.array(tensor.contents.uint64_contents)
        elif tensor_type == "FP32":
            return np.array(tensor.contents.fp32_contents)
        elif tensor_type == "FP64":
            return np.array(tensor.contents.fp64_contents)
        elif tensor_type == "BYTES":
            return np.array([bytes(b) for b in tensor.contents.bytes_contents])
        else:
            raise ValueError(f"Unsupported tensor type: {tensor_type}")

    @staticmethod
    def _get_tensor_type(datatype: str) -> str:
        """Convert datatype string to tensor type."""
        datatype_mapping = {
            "BOOL": "BOOL",
            "INT8": "INT8",
            "INT16": "INT16",
            "INT32": "INT32",
            "INT64": "INT64",
            "UINT8": "UINT8",
            "UINT16": "UINT16",
            "UINT32": "UINT32",
            "UINT64": "UINT64",
            "FP32": "FP32",
            "FP64": "FP64",
            "BYTES": "BYTES",
            "STRING": "BYTES",
            # numpy/Python types
            "float32": "FP32",
            "float64": "FP64",
            "int32": "INT32",
            "int64": "INT64",
        }

        return datatype_mapping.get(datatype.upper(), datatype.upper())

    @staticmethod
    def _has_tensor_data(tensor) -> bool:
        """Check if a tensor has data."""
        return tensor.contents and (
            tensor.contents.bool_contents
            or tensor.contents.int_contents
            or tensor.contents.int64_contents
            or tensor.contents.uint_contents
            or tensor.contents.uint64_contents
            or tensor.contents.fp32_contents
            or tensor.contents.fp64_contents
            or tensor.contents.bytes_contents
        )

    @staticmethod
    def _extract_input_tensors(
        input_request: ModelInferRequest,
    ) -> Tuple[List[np.ndarray], List[str], List[bool]]:
        """Extract features, names and synthetic flags from input request."""
        input_features = []
        input_names = []
        synthetic_flags = []

        for tensor in input_request.inputs:
            is_synthetic = ModelMeshPayloadParser._is_synthetic_tensor(tensor)
            tensor_type = ModelMeshPayloadParser._get_tensor_type(tensor.datatype)
            shape = list(tensor.shape)

            data = ModelMeshPayloadParser._get_tensor_data(tensor, tensor_type, input_request)

            if len(shape) > 1:
                data = data.reshape(shape)

            input_features.append(data)
            input_names.append(tensor.name)
            synthetic_flags.append(is_synthetic)

        return input_features, input_names, synthetic_flags

    @staticmethod
    def _is_synthetic_tensor(tensor) -> bool:
        """Check if a tensor is synthetic (has bias_ignore parameter set to 'true')."""
        if not tensor.parameters:
            return False

        param_name = ModelMeshPayloadParser.BIAS_IGNORE_PARAM
        if param_name not in tensor.parameters:
            return False

        param = tensor.parameters[param_name]
        return hasattr(param, "string_param") and param.string_param == "true"

    @staticmethod
    def _get_tensor_data(tensor, tensor_type: str, request_obj: Any) -> np.ndarray:
        """Extract data from a tensor."""
        if ModelMeshPayloadParser._has_tensor_data(tensor):
            return ModelMeshPayloadParser._extract_tensor_data(tensor, tensor_type)
        elif hasattr(request_obj, "raw_input_contents") and request_obj.raw_input_contents:
            raise NotImplementedError("Raw input contents parsing not yet implemented")
        elif hasattr(request_obj, "raw_output_contents") and request_obj.raw_output_contents:
            raise NotImplementedError("Raw output contents parsing not yet implemented")
        else:
            raise ValueError(f"No data found in tensor {tensor.name}")

    @staticmethod
    def _extract_output_tensors(
        output_response: ModelInferResponse,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Extract features and names from output response."""
        output_features = []
        output_names = []

        for tensor in output_response.outputs:
            tensor_type = ModelMeshPayloadParser._get_tensor_type(tensor.datatype)
            shape = list(tensor.shape)

            data = ModelMeshPayloadParser._get_tensor_data(tensor, tensor_type, output_response)

            if len(shape) > 1:
                data = data.reshape(shape)

            output_features.append(data)
            output_names.append(tensor.name)

        return output_features, output_names

    @staticmethod
    def _build_dataframe_rows(
        input_features: List[np.ndarray],
        input_names: List[str],
        output_features: List[np.ndarray],
        output_names: List[str],
        synthetic_flags: List[bool],
        id: str,
        model_id: str,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Build DataFrame rows from extracted features."""
        rows = []
        for i in range(batch_size):
            row = {}

            # Inputs
            for j, name in enumerate(input_names):
                data = input_features[j]
                row[name] = data[i] if i < len(data) else None

            # Outputs
            for j, name in enumerate(output_names):
                data = output_features[j]
                row[f"output_{name}"] = data[i] if i < len(data) else None

            row["id"] = id
            row["model_id"] = ModelMeshPayloadParser.standardize_model_id(model_id)
            row["synthetic"] = any(synthetic_flags)

            rows.append(row)

        return rows

    @staticmethod
    def payloads_to_dataframe(
        input_payload: PartialPayload,
        output_payload: PartialPayload,
        id: str,
        model_id: str,
    ) -> pd.DataFrame:
        """
        Convert both input and output payloads to a pandas DataFrame.
        """
        # Parse payloads
        input_request = ModelMeshPayloadParser.parse_input_payload(input_payload)
        output_response = ModelMeshPayloadParser.parse_output_payload(output_payload)

        # Extract input and output tensor data
        input_features, input_names, synthetic_flags = ModelMeshPayloadParser._extract_input_tensors(input_request)
        output_features, output_names = ModelMeshPayloadParser._extract_output_tensors(output_response)

        # Determine batch size
        batch_size = 1
        if input_features and len(input_features[0].shape) > 0:
            batch_size = input_features[0].shape[0]

        # Build DataFrame rows
        rows = ModelMeshPayloadParser._build_dataframe_rows(
            input_features,
            input_names,
            output_features,
            output_names,
            synthetic_flags,
            id,
            model_id,
            batch_size,
        )

        # Create DataFrame and set attributes
        df = pd.DataFrame(rows)
        df.attrs["input_tensor_name"] = input_names[0] if input_names else None
        df.attrs["output_tensor_name"] = output_names[0] if output_names else None

        return df
