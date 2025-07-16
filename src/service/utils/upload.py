import logging
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException

from src.service.constants import (
    INPUT_SUFFIX,
    METADATA_SUFFIX,
    OUTPUT_SUFFIX,
    TRUSTYAI_TAG_PREFIX,
)
from src.service.data.modelmesh_parser import ModelMeshPayloadParser
from src.service.data.storage import get_storage_interface
from src.service.utils import list_utils
from src.endpoints.consumer.consumer_endpoint import process_payload


logger = logging.getLogger(__name__)


METADATA_STRING_MAX_LENGTH = 100


class KServeDataAdapter:
    """
    Convert upload tensors to consumer endpoint format.
    """

    def __init__(self, tensor_dict: Dict[str, Any], numpy_array: np.ndarray):
        """Initialize adapter with validated data."""
        self._name = tensor_dict.get("name", "unknown")
        self._shape = tensor_dict.get("shape", [])
        self._datatype = tensor_dict.get("datatype", "FP64")
        self._data = numpy_array  # Keep numpy array intact

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def datatype(self) -> str:
        return self._datatype

    @property
    def data(self) -> np.ndarray:
        """Returns numpy array with .shape attribute as expected by consumer endpoint."""
        return self._data


class ConsumerEndpointAdapter:
    """
    Consumer endpoint's expected structure.
    """

    def __init__(self, adapted_tensors: List[KServeDataAdapter]):
        self.tensors = adapted_tensors
        self.id = f"upload_request_{uuid.uuid4().hex[:8]}"


async def process_upload_request(payload: Any) -> Dict[str, str]:
    """
    Process complete upload request with validation and data handling.
    """
    try:
        model_name = ModelMeshPayloadParser.standardize_model_id(payload.model_name)
        if payload.data_tag:
            error = validate_data_tag(payload.data_tag)
            if error:
                raise HTTPException(400, error)
        inputs = payload.request.get("inputs", [])
        outputs = payload.response.get("outputs", []) if payload.response else []
        if not inputs:
            raise HTTPException(400, "Missing input tensors")
        if payload.is_ground_truth and not outputs:
            raise HTTPException(400, "Ground truth uploads require output tensors")

        input_arrays, input_names, _, execution_ids = process_tensors_using_kserve_logic(inputs)
        if outputs:
            output_arrays, output_names, _, _ = process_tensors_using_kserve_logic(outputs)
        else:
            output_arrays, output_names = [], []
        error = validate_input_shapes(input_arrays, input_names)
        if error:
            raise HTTPException(400, f"One or more errors in input tensors: {error}")
        if payload.is_ground_truth:
            return await _process_ground_truth_data(
                model_name, input_arrays, input_names, output_arrays, output_names, execution_ids
            )
        else:
            return await _process_regular_data(
                model_name, input_arrays, input_names, output_arrays, output_names, execution_ids, payload.data_tag
            )
    except ProcessingError as e:
        raise HTTPException(400, str(e))
    except ValidationError as e:
        raise HTTPException(400, str(e))


async def _process_ground_truth_data(
    model_name: str,
    input_arrays: List[np.ndarray],
    input_names: List[str],
    output_arrays: List[np.ndarray],
    output_names: List[str],
    execution_ids: Optional[List[str]],
) -> Dict[str, str]:
    """Process ground truth data upload."""
    if not execution_ids:
        raise HTTPException(400, "Ground truth requires execution IDs")
    result = await handle_ground_truths(
        model_name,
        input_arrays,
        input_names,
        output_arrays,
        output_names,
        [sanitize_id(id) for id in execution_ids],
    )
    if not result.success:
        raise HTTPException(400, result.message)
    result_data = result.data
    if result_data is None:
        raise HTTPException(500, "Ground truth processing failed")
    gt_name = f"{model_name}_ground_truth"
    storage_interface = get_storage_interface()
    await storage_interface.write_data(gt_name + OUTPUT_SUFFIX, result_data["outputs"], result_data["output_names"])
    await storage_interface.write_data(
        gt_name + METADATA_SUFFIX,
        result_data["metadata"],
        result_data["metadata_names"],
    )
    logger.info(f"Ground truth data saved for model: {model_name}")
    return {"message": result.message}


async def _process_regular_data(
    model_name: str,
    input_arrays: List[np.ndarray],
    input_names: List[str],
    output_arrays: List[np.ndarray],
    output_names: List[str],
    execution_ids: Optional[List[str]],
    data_tag: Optional[str],
) -> Dict[str, str]:
    """Process regular model data upload."""
    n_rows = input_arrays[0].shape[0]
    exec_ids = execution_ids or [str(uuid.uuid4()) for _ in range(n_rows)]
    input_data = _flatten_tensor_data(input_arrays, n_rows)
    output_data = _flatten_tensor_data(output_arrays, n_rows)
    metadata, metadata_cols = _create_metadata(exec_ids, model_name, data_tag)
    await save_model_data(
        model_name,
        np.array(input_data),
        input_names,
        np.array(output_data),
        output_names,
        metadata,
        metadata_cols,
    )
    logger.info(f"Regular data saved for model: {model_name}, rows: {n_rows}")
    return {"message": f"{n_rows} datapoints added to {model_name}"}


def _flatten_tensor_data(arrays: List[np.ndarray], n_rows: int) -> List[List[Any]]:
    """
    Flatten tensor arrays into row-based format for storage.
    """

    def flatten_row(arrays: List[np.ndarray], row: int) -> List[Any]:
        """Flatten arrays for a single row."""
        return [x for arr in arrays for x in (arr[row].flatten() if arr.ndim > 1 else [arr[row]])]

    return [flatten_row(arrays, i) for i in range(n_rows)]


def _create_metadata(
    execution_ids: List[str], model_name: str, data_tag: Optional[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Create metadata array for model data storage.
    """
    current_timestamp = datetime.now().isoformat()
    metadata_cols = ["ID", "MODEL_ID", "TIMESTAMP", "TAG"]
    metadata_rows = [
        [
            str(eid),
            str(model_name),
            str(current_timestamp),
            str(data_tag or ""),
        ]
        for eid in execution_ids
    ]
    _validate_metadata_lengths(metadata_rows, metadata_cols)
    metadata = np.array(metadata_rows, dtype=f"<U{METADATA_STRING_MAX_LENGTH}")
    return metadata, metadata_cols


def _validate_metadata_lengths(metadata_rows: List[List[str]], column_names: List[str]) -> None:
    """
    Validate that all metadata values fit within the defined string length limit.
    """
    for row_idx, row in enumerate(metadata_rows):
        for col_idx, value in enumerate(row):
            value_str = str(value)
            if len(value_str) > METADATA_STRING_MAX_LENGTH:
                col_name = column_names[col_idx] if col_idx < len(column_names) else f"column_{col_idx}"
                raise ValidationError(
                    f"Metadata field '{col_name}' in row {row_idx} exceeds maximum length "
                    f"of {METADATA_STRING_MAX_LENGTH} characters (got {len(value_str)} chars): "
                    f"'{value_str[:50]}{'...' if len(value_str) > 50 else ''}'"
                )


class ValidationError(Exception):
    """Validation errors."""

    pass


class ProcessingError(Exception):
    """Processing errors."""

    pass


@dataclass
class GroundTruthValidationResult:
    """Result of ground truth validation."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


TYPE_MAP = {
    np.int64: "Long",
    np.int32: "Integer",
    np.float32: "Float",
    np.float64: "Double",
    np.bool_: "Boolean",
    int: "Long",
    float: "Double",
    bool: "Boolean",
    str: "String",
}


def get_type_name(val: Any) -> str:
    """Get Java-style type name for a value (used in ground truth validation)."""
    if hasattr(val, "dtype"):
        return TYPE_MAP.get(val.dtype.type, "String")
    return TYPE_MAP.get(type(val), "String")


def sanitize_id(execution_id: str) -> str:
    """Sanitize execution ID."""
    return str(execution_id).strip()


def extract_row_data(arrays: List[np.ndarray], row_index: int) -> List[Any]:
    """Extract data from arrays for a specific row."""
    row_data = []
    for arr in arrays:
        if arr.ndim > 1:
            row_data.extend(arr[row_index].flatten())
        else:
            row_data.append(arr[row_index])
    return row_data


def process_tensors_using_kserve_logic(
    tensors: List[Dict[str, Any]],
) -> Tuple[List[np.ndarray], List[str], List[str], Optional[List[str]]]:
    """
    Process tensor data using consumer endpoint logic via clean adapter pattern.
    """
    if not tensors:
        return [], [], [], None
    validation_errors = _validate_tensor_inputs(tensors)
    if validation_errors:
        error_message = "One or more errors occurred: " + ". ".join(validation_errors)
        raise HTTPException(400, error_message)
    adapted_tensors = []
    execution_ids = None
    datatypes = []
    for tensor in tensors:
        if execution_ids is None:
            execution_ids = tensor.get("execution_ids")
        numpy_array = _convert_tensor_to_numpy(tensor)
        adapter = KServeDataAdapter(tensor, numpy_array)
        adapted_tensors.append(adapter)
        datatypes.append(adapter.datatype)
    try:
        adapter_payload = ConsumerEndpointAdapter(adapted_tensors)
        tensor_array, column_names = process_payload(adapter_payload, lambda payload: payload.tensors)
        arrays, all_names = _convert_consumer_results_to_upload_format(tensor_array, column_names, adapted_tensors)
        return arrays, all_names, datatypes, execution_ids
    except Exception as e:
        logger.error(f"Consumer endpoint processing failed: {e}")
        raise HTTPException(400, f"Tensor processing error: {str(e)}")


def _validate_tensor_inputs(tensors: List[Dict[str, Any]]) -> List[str]:
    """Validate tensor inputs and return list of error messages."""
    errors = []
    tensor_names = [tensor.get("name", f"tensor_{i}") for i, tensor in enumerate(tensors)]
    if len(tensor_names) != len(set(tensor_names)):
        errors.append("Input tensors must have unique names")
    shapes = [tensor.get("shape", []) for tensor in tensors]
    if len(shapes) > 1:
        first_dims = [shape[0] if shape else 0 for shape in shapes]
        if len(set(first_dims)) > 1:
            errors.append(f"Input tensors must have consistent first dimension. Found: {first_dims}")
    return errors


def _convert_tensor_to_numpy(tensor: Dict[str, Any]) -> np.ndarray:
    """Convert tensor dictionary to numpy array with proper dtype."""
    raw_data = tensor.get("data", [])

    if list_utils.contains_non_numeric(raw_data):
        return np.array(raw_data, dtype="O")
    dtype_map = {"INT64": np.int64, "INT32": np.int32, "FP32": np.float32, "FP64": np.float64, "BOOL": np.bool_}
    datatype = tensor.get("datatype", "FP64")
    np_dtype = dtype_map.get(datatype, np.float64)
    return np.array(raw_data, dtype=np_dtype)


def _convert_consumer_results_to_upload_format(
    tensor_array: np.ndarray, column_names: List[str], adapted_tensors: List[KServeDataAdapter]
) -> Tuple[List[np.ndarray], List[str]]:
    """Convert consumer endpoint results back to upload format."""
    if len(adapted_tensors) == 1:
        # Single tensor case
        return [tensor_array], column_names
    arrays = []
    all_names = []
    col_start = 0
    for adapter in adapted_tensors:
        if len(adapter.shape) > 1:
            n_cols = adapter.shape[1]
            tensor_names = [f"{adapter.name}-{i}" for i in range(n_cols)]
        else:
            n_cols = 1
            tensor_names = [adapter.name]
        if tensor_array.ndim == 2:
            tensor_data = tensor_array[:, col_start : col_start + n_cols]
        else:
            tensor_data = tensor_array[col_start : col_start + n_cols]
        arrays.append(tensor_data)
        all_names.extend(tensor_names)
        col_start += n_cols
    return arrays, all_names


def validate_input_shapes(input_arrays: List[np.ndarray], input_names: List[str]) -> Optional[str]:
    """Validate input array shapes and names - collect ALL errors."""
    if not input_arrays:
        return None
    errors = []
    if len(set(input_names)) != len(input_names):
        errors.append("Input tensors must have unique names")
    first_dim = input_arrays[0].shape[0]
    for i, arr in enumerate(input_arrays[1:], 1):
        if arr.shape[0] != first_dim:
            errors.append(
                f"Input tensor '{input_names[i]}' has first dimension {arr.shape[0]}, "
                f"which doesn't match the first dimension {first_dim} of '{input_names[0]}'"
            )
    if errors:
        return ". ".join(errors) + "."
    return None


def validate_data_tag(tag: str) -> Optional[str]:
    """Validate data tag format and content."""
    if not tag:
        return None
    if tag.startswith(TRUSTYAI_TAG_PREFIX):
        return (
            f"The tag prefix '{TRUSTYAI_TAG_PREFIX}' is reserved for internal TrustyAI use only. "
            f"Provided tag '{tag}' violates this restriction."
        )
    return None


class GroundTruthValidator:
    """Ground truth validator."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.id_to_row: Dict[str, int] = {}
        self.inputs: Optional[np.ndarray] = None
        self.outputs: Optional[np.ndarray] = None
        self.metadata: Optional[np.ndarray] = None

    async def initialize(self) -> None:
        """Load existing data."""
        storage_interface = get_storage_interface()
        self.inputs, _ = await storage_interface.read_data(self.model_name + INPUT_SUFFIX)
        self.outputs, _ = await storage_interface.read_data(self.model_name + OUTPUT_SUFFIX)
        self.metadata, _ = await storage_interface.read_data(self.model_name + METADATA_SUFFIX)
        metadata_cols = await storage_interface.get_original_column_names(self.model_name + METADATA_SUFFIX)
        id_col = next((i for i, name in enumerate(metadata_cols) if name.upper() == "ID"), 0)
        if self.metadata is not None:
            for j, row in enumerate(self.metadata):
                id_val = row[id_col]
                self.id_to_row[str(id_val)] = j

    def find_row(self, exec_id: str) -> Optional[int]:
        """Find row index for execution ID."""
        return self.id_to_row.get(str(exec_id))

    async def validate_data(
        self,
        exec_id: str,
        uploaded_inputs: List[Any],
        uploaded_outputs: List[Any],
        row_idx: int,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Validate inputs and outputs."""
        if self.inputs is None or self.outputs is None:
            return f"ID={exec_id} no existing data found"
        existing_inputs = self.inputs[row_idx]
        existing_outputs = self.outputs[row_idx]
        for i, (existing, uploaded) in enumerate(zip(existing_inputs[:3], uploaded_inputs[:3])):
            if hasattr(existing, "dtype"):
                print(
                    f"  Input {i}: existing.dtype={existing.dtype}, uploaded.dtype={getattr(uploaded, 'dtype', 'no dtype')}"
                )
            print(f"  Input {i}: existing={existing}, uploaded={uploaded}")
        for i, (existing, uploaded) in enumerate(zip(existing_outputs[:2], uploaded_outputs[:2])):
            if hasattr(existing, "dtype"):
                print(
                    f"  Output {i}: existing.dtype={existing.dtype}, uploaded.dtype={getattr(uploaded, 'dtype', 'no dtype')}"
                )
            print(f"  Output {i}: existing={existing}, uploaded={uploaded}")
        if len(existing_inputs) != len(uploaded_inputs):
            return f"ID={exec_id} input shapes do not match. Observed inputs have length={len(existing_inputs)} while uploaded inputs have length={len(uploaded_inputs)}"
        for i, (existing, uploaded) in enumerate(zip(existing_inputs, uploaded_inputs)):
            existing_type = get_type_name(existing)
            uploaded_type = get_type_name(uploaded)
            print(f"  Input {i}: existing_type='{existing_type}', uploaded_type='{uploaded_type}'")
            if existing_type != uploaded_type:
                return f"ID={exec_id} input type mismatch at position {i + 1}: Class={existing_type} != Class={uploaded_type}"
            if existing != uploaded:
                return f"ID={exec_id} inputs are not identical: value mismatch at position {i + 1}"
        if len(existing_outputs) != len(uploaded_outputs):
            return f"ID={exec_id} output shapes do not match. Observed outputs have length={len(existing_outputs)} while uploaded ground-truths have length={len(uploaded_outputs)}"
        for i, (existing, uploaded) in enumerate(zip(existing_outputs, uploaded_outputs)):
            existing_type = get_type_name(existing)
            uploaded_type = get_type_name(uploaded)
            print(f"  Output {i}: existing_type='{existing_type}', uploaded_type='{uploaded_type}'")
            if existing_type != uploaded_type:
                return f"ID={exec_id} output type mismatch at position {i + 1}: Class={existing_type} != Class={uploaded_type}"
        return None


async def handle_ground_truths(
    model_name: str,
    input_arrays: List[np.ndarray],
    input_names: List[str],
    output_arrays: List[np.ndarray],
    output_names: List[str],
    execution_ids: List[str],
    config: Optional[Any] = None,
) -> GroundTruthValidationResult:
    """Handle ground truth validation."""
    if not execution_ids:
        return GroundTruthValidationResult(success=False, message="No execution IDs provided.")
    storage_interface = get_storage_interface()
    if not await storage_interface.dataset_exists(model_name + INPUT_SUFFIX):
        return GroundTruthValidationResult(success=False, message=f"Model {model_name} not found.")
    validator = GroundTruthValidator(model_name)
    await validator.initialize()
    errors = []
    valid_outputs = []
    valid_metadata = []
    n_rows = input_arrays[0].shape[0] if input_arrays else 0
    for i, exec_id in enumerate(execution_ids):
        if i >= n_rows:
            errors.append(f"ID={exec_id} index out of bounds")
            continue
        row_idx = validator.find_row(exec_id)
        if row_idx is None:
            errors.append(f"ID={exec_id} not found")
            continue
        uploaded_inputs = extract_row_data(input_arrays, i)
        uploaded_outputs = extract_row_data(output_arrays, i)
        error = await validator.validate_data(exec_id, uploaded_inputs, uploaded_outputs, row_idx)
        if error:
            errors.append(error)
            continue
        valid_outputs.append(uploaded_outputs)
        valid_metadata.append([exec_id])
    if errors:
        return GroundTruthValidationResult(
            success=False,
            message="Found fatal mismatches between uploaded data and recorded inference data:\n"
            + "\n".join(errors[:5]),
            errors=errors,
        )
    if not valid_outputs:
        return GroundTruthValidationResult(success=False, message="No valid ground truths found.")
    return GroundTruthValidationResult(
        success=True,
        message=f"{len(valid_outputs)} ground truths added.",
        data={
            "outputs": np.array(valid_outputs),
            "output_names": output_names,
            "metadata": np.array(valid_metadata),
            "metadata_names": ["ID"],
        },
    )


async def save_model_data(
    model_name: str,
    input_data: np.ndarray,
    input_names: List[str],
    output_data: np.ndarray,
    output_names: List[str],
    metadata_data: np.ndarray,
    metadata_names: List[str],
) -> Dict[str, Any]:
    """Save model data to storage."""
    storage_interface = get_storage_interface()
    await storage_interface.write_data(model_name + INPUT_SUFFIX, input_data, input_names)
    await storage_interface.write_data(model_name + OUTPUT_SUFFIX, output_data, output_names)
    await storage_interface.write_data(model_name + METADATA_SUFFIX, metadata_data, metadata_names)
    logger.info(f"Saved model data for {model_name}: {len(input_data)} rows")
    return {
        "model_name": model_name,
        "rows": len(input_data),
    }