"""
ModelMesh protobuf testing utils.
"""

import base64
import random
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np

try:
    from src.proto.grpc_predict_v2_pb2 import (
        ModelInferRequest,
        ModelInferResponse,
        InferParameter,
        InferTensorContents,
    )
except ImportError:
    print("Warning: Protobuf classes not available. Run generate_protos.py first.")


class ModelMeshTestData:
    """
    ModelMesh test data for testing the protobuf parser.
    """

    @staticmethod
    def create_infer_parameter(value: Union[bool, int, str]) -> InferParameter:
        """Create an InferParameter with the correct type based on value."""
        param = InferParameter()
        if isinstance(value, bool):
            param.bool_param = value
        elif isinstance(value, int):
            param.int64_param = value
        elif isinstance(value, str):
            param.string_param = value
        return param

    @staticmethod
    def generate_data(
        rows: int, cols: int, datatype: str, offset: int = 0
    ) -> Tuple[np.ndarray, Any]:
        """
        Create test data based on the datatype.
        Returns the NumPy array and the corresponding value for InferTensorContents.
        """
        if datatype == "BOOL":
            data = np.array(
                [(i + offset) % 2 == 0 for i in range(rows * cols)]
            ).reshape(rows, cols)
            tensor_data = data.flatten().tolist()
            return data, tensor_data
        elif datatype in {"INT8", "INT16", "INT32"}:
            data = np.array(
                [i + offset for i in range(rows * cols)], dtype=np.int32
            ).reshape(rows, cols)
            tensor_data = data.flatten().tolist()
            return data, tensor_data
        elif datatype == "INT64":
            data = np.array(
                [i + offset for i in range(rows * cols)], dtype=np.int64
            ).reshape(rows, cols)
            tensor_data = data.flatten().tolist()
            return data, tensor_data
        elif datatype == "FP32":
            data = np.array(
                [float(i + offset) for i in range(rows * cols)], dtype=np.float32
            ).reshape(rows, cols)
            tensor_data = data.flatten().tolist()
            return data, tensor_data
        elif datatype == "FP64":
            data = np.array(
                [(i + offset) / 2.0 for i in range(rows * cols)], dtype=np.float64
            ).reshape(rows, cols)
            tensor_data = data.flatten().tolist()
            return data, tensor_data
        elif datatype == "BYTES":
            data = np.array(
                [str(i + offset).encode() for i in range(rows * cols)]
            ).reshape(rows, cols)
            tensor_data = data.flatten().tolist()
            return data, tensor_data
        else:
            raise ValueError(f"Unsupported datatype: {datatype}")

    @staticmethod
    def create_tensor_contents(datatype: str, data: Any) -> InferTensorContents:
        """InferTensorContents with the appropriate field based on datatype."""
        contents = InferTensorContents()

        if datatype == "BOOL":
            contents.bool_contents.extend(data)
        elif datatype in {"INT8", "INT16", "INT32"}:
            contents.int_contents.extend(data)
        elif datatype == "INT64":
            contents.int64_contents.extend(data)
        elif datatype in {"UINT8", "UINT16", "UINT32"}:
            contents.uint_contents.extend(data)
        elif datatype == "UINT64":
            contents.uint64_contents.extend(data)
        elif datatype == "FP32":
            contents.fp32_contents.extend(data)
        elif datatype == "FP64":
            contents.fp64_contents.extend(data)
        elif datatype == "BYTES":
            contents.bytes_contents.extend(data)

        return contents

    @staticmethod
    def generate_input_tensor(
        name: str,
        rows: int,
        cols: int,
        datatype: str,
        offset: int = 0,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ModelInferRequest.InferInputTensor, np.ndarray]:
        """Generate an input tensor with test data."""
        tensor = ModelInferRequest.InferInputTensor()
        tensor.name = name
        tensor.datatype = datatype
        tensor.shape.extend([rows, cols])

        if parameters:
            for key, value in parameters.items():
                tensor.parameters[key].CopyFrom(
                    ModelMeshTestData.create_infer_parameter(value)
                )

        data_np, tensor_data = ModelMeshTestData.generate_data(
            rows, cols, datatype, offset
        )

        tensor.contents.CopyFrom(
            ModelMeshTestData.create_tensor_contents(datatype, tensor_data)
        )

        return tensor, data_np

    @staticmethod
    def generate_output_tensor(
        name: str, rows: int, cols: int, datatype: str, offset: int = 0
    ) -> Tuple[ModelInferResponse.InferOutputTensor, np.ndarray]:
        """Generate an output tensor with test data."""
        tensor = ModelInferResponse.InferOutputTensor()
        tensor.name = name
        tensor.datatype = datatype
        tensor.shape.extend([rows, cols])

        data_np, tensor_data = ModelMeshTestData.generate_data(
            rows, cols, datatype, offset
        )

        tensor.contents.CopyFrom(
            ModelMeshTestData.create_tensor_contents(datatype, tensor_data)
        )

        return tensor, data_np

    @staticmethod
    def generate_model_infer_request(
        model_name: str,
        input_tensors: List[Tuple[str, int, int, str, int, Optional[Dict[str, Any]]]],
    ) -> Tuple[ModelInferRequest, Dict[str, np.ndarray]]:
        """
        Generate a ModelInferRequest with the specified input tensors.
        """
        request = ModelInferRequest()
        request.model_name = model_name
        request.id = f"test-request-{random.randint(1000, 9999)}"

        data_dict = {}

        for tensor_spec in input_tensors:
            name, rows, cols, datatype, offset, params = tensor_spec
            tensor, data_np = ModelMeshTestData.generate_input_tensor(
                name, rows, cols, datatype, offset, params
            )
            request.inputs.append(tensor)
            data_dict[name] = data_np

        return request, data_dict

    @staticmethod
    def generate_model_infer_response(
        model_name: str,
        model_version: str,
        request_id: str,
        output_tensors: List[Tuple[str, int, int, str, int]],
    ) -> Tuple[ModelInferResponse, Dict[str, np.ndarray]]:
        """
        Generate a ModelInferResponse with the specified output tensors.
        """
        response = ModelInferResponse()
        response.model_name = model_name
        response.model_version = model_version
        response.id = request_id

        data_dict = {}

        for tensor_spec in output_tensors:
            name, rows, cols, datatype, offset = tensor_spec
            tensor, data_np = ModelMeshTestData.generate_output_tensor(
                name, rows, cols, datatype, offset
            )
            response.outputs.append(tensor)
            data_dict[name] = data_np

        return response, data_dict

    @staticmethod
    def generate_test_payloads(
        model_name: str,
        input_tensor_specs: List[
            Tuple[str, int, int, str, int, Optional[Dict[str, Any]]]
        ],
        output_tensor_specs: List[Tuple[str, int, int, str, int]],
    ) -> Tuple[dict, dict, dict, dict]:
        """
        Generate test input and output payloads for ModelMesh testing.
        """
        request, input_data_dict = ModelMeshTestData.generate_model_infer_request(
            model_name, input_tensor_specs
        )

        response, output_data_dict = ModelMeshTestData.generate_model_infer_response(
            f"{model_name}__isvc-123456",  # Add ModelMesh suffix
            "1",
            request.id,
            output_tensor_specs,
        )

        request_bytes = request.SerializeToString()
        response_bytes = response.SerializeToString()

        request_b64 = base64.b64encode(request_bytes).decode("utf-8")
        response_b64 = base64.b64encode(response_bytes).decode("utf-8")

        input_payload = {"data": request_b64, "metadata": {"source": "test"}}

        output_payload = {"data": response_b64, "metadata": {"source": "test"}}

        return input_payload, output_payload, input_data_dict, output_data_dict
