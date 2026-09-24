"""Bounded, protocol-only KServe V2 JSON encoding and decoding."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Integral
from typing import Literal, NoReturn, Protocol, cast

import numpy as np

from .model_provider import (
    PredictionMetadata,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)

_NUMERIC_DATATYPES = frozenset(
    {
        "BOOL",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "FP16",
        "FP32",
        "FP64",
    }
)
_MAX_REQUEST_ELEMENTS = 10_000_000
_MAX_RESPONSE_ELEMENTS = 10_000_000
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_MATRIX_RANK = 2
_HTTP_SUCCESS_MIN = 200
_HTTP_REDIRECT = 300
_HTTP_SERVER_ERROR = 500

_Direction = Literal["request", "response"]


class BoundedResponse(Protocol):
    """Minimum response boundary required by the codec."""

    status_code: int
    content: bytes

    def json(self) -> object:
        """Decode a response body that has already been byte-bounded."""
        ...


def _invalid_request(message: str) -> NoReturn:
    """Raise the canonical invalid-request error."""
    raise ProviderInvalidRequestError(message)


def _invalid_response(message: str) -> NoReturn:
    """Raise the canonical invalid-response error."""
    raise ProviderInvalidResponseError(message)


def _unsupported_model(message: str) -> NoReturn:
    """Raise the canonical unsupported-model error."""
    raise ProviderUnsupportedModelError(message)


def _parse_shape(value: object) -> tuple[int, ...]:
    """Parse a JSON shape without coercing malformed dimensions."""
    if not isinstance(value, list):
        _invalid_response("tensor shape must be a JSON list")
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, Integral)
        for dimension in value
    ):
        _invalid_response("tensor dimensions must be integers")
    return tuple(int(dimension) for dimension in value)


def _validate_metadata_shape(shape: tuple[int, ...]) -> None:
    """Validate the supported flat tabular metadata shape."""
    if not shape or len(shape) > _MATRIX_RANK:
        _unsupported_model("Only flat rank-one or rank-two tensors are supported")
    if any(dimension == 0 or dimension < -1 for dimension in shape):
        _unsupported_model("Tensor dimensions must be positive or -1")
    if any(dimension == -1 for dimension in shape[1:]):
        _unsupported_model("Only a leading dynamic batch dimension is supported")
    if len(shape) == _MATRIX_RANK and shape[0] not in {-1, 1}:
        _unsupported_model("Fixed model batches other than one are unsupported")


def _validate_numeric_datatype(datatype: object, *, direction: _Direction) -> None:
    """Require a supported numeric datatype on direct codec calls as well."""
    if isinstance(datatype, str) and datatype in _NUMERIC_DATATYPES:
        return
    message = "Only numeric tensors are supported"
    if direction == "request":
        _invalid_request(message)
    _invalid_response(message)


def _tensor_list(
    payload: Mapping[str, object], name: str, *, allow_missing: bool
) -> list[Mapping[str, object]]:
    """Read a metadata or response tensor list without accepting loose coercions."""
    if name not in payload and allow_missing:
        return []
    value = payload.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        _invalid_response(f"model {name} must be a list of tensors")
    return cast("list[Mapping[str, object]]", value)


def _select_tensor(
    tensors: list[Mapping[str, object]],
    selector: str | None,
    label: str,
) -> Mapping[str, object]:
    """Select one tensor while preserving selector and upstream ambiguity errors."""
    selected = [
        tensor
        for tensor in tensors
        if selector is None or tensor.get("name") == selector
    ]
    if not selected:
        if selector is None:
            _unsupported_model(f"Model {label} metadata is missing")
        _invalid_request(f"Requested model {label} tensor was not found")
    if len(selected) != 1:
        _unsupported_model(f"Model {label} metadata is ambiguous")
    return selected[0]


def _validate_advertised_versions(
    payload: Mapping[str, object], model_version: str | None
) -> None:
    """Validate an optional model version advertisement and requested selector."""
    if "versions" not in payload:
        return
    versions = payload["versions"]
    if not isinstance(versions, list) or any(
        not isinstance(version, str) or not version for version in versions
    ):
        _unsupported_model("Model metadata versions must be a list")
    if model_version is not None and model_version not in versions:
        _invalid_request("Requested model version was not found")


def parse_metadata(
    payload: object,
    model_name: str,
    model_version: str | None = None,
    input_name: str | None = None,
    output_name: str | None = None,
) -> PredictionMetadata:
    """Parse and validate one selected numeric KServe V2 model contract."""
    if not isinstance(payload, Mapping):
        _invalid_response("model metadata must be a JSON object")
    if payload.get("name") != model_name:
        _unsupported_model("Model metadata name does not match the request")
    _validate_advertised_versions(payload, model_version)

    inputs = _tensor_list(payload, "inputs", allow_missing=True)
    outputs = _tensor_list(payload, "outputs", allow_missing=True)
    if len(inputs) != 1:
        _unsupported_model(
            "Model input metadata is ambiguous or has multiple required inputs"
        )
    input_tensor = _select_tensor(inputs, input_name, "input")
    output_tensor = _select_tensor(outputs, output_name, "output")

    input_datatype = input_tensor.get("datatype")
    output_datatype = output_tensor.get("datatype")
    if (
        not isinstance(input_datatype, str)
        or input_datatype not in _NUMERIC_DATATYPES
        or not isinstance(output_datatype, str)
        or output_datatype not in _NUMERIC_DATATYPES
    ):
        _unsupported_model("Only numeric tensors are supported")

    selected_input_name = input_tensor.get("name")
    selected_output_name = output_tensor.get("name")
    if not isinstance(selected_input_name, str) or not selected_input_name:
        _invalid_response("model tensor names must be strings")
    if not isinstance(selected_output_name, str) or not selected_output_name:
        _invalid_response("model tensor names must be strings")

    input_shape = _parse_shape(input_tensor.get("shape", []))
    output_shape = _parse_shape(output_tensor.get("shape", []))
    _validate_metadata_shape(input_shape)
    _validate_metadata_shape(output_shape)

    return PredictionMetadata(
        input_name=selected_input_name,
        output_name=selected_output_name,
        input_datatype=input_datatype,
        output_datatype=output_datatype,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def _validate_numeric_values(
    values: np.ndarray, datatype: str, *, direction: _Direction
) -> None:
    """Validate BOOL and integer ranges for request and response directions."""
    is_request = direction == "request"
    if datatype == "BOOL":
        valid = np.all(np.isin(values, [0, 1]))
        if not valid:
            message = (
                "Boolean model inputs must be zero or one"
                if is_request
                else "Boolean model outputs are invalid"
            )
            (_invalid_request(message) if is_request else _invalid_response(message))
        return

    if datatype.startswith(("INT", "UINT")):
        if values.dtype.kind == "f" and not np.equal(values, np.floor(values)).all():
            message = (
                "Integer model inputs must be integral"
                if is_request
                else "Integer model outputs are not integral"
            )
            (_invalid_request(message) if is_request else _invalid_response(message))
        bits = int(datatype.removeprefix("UINT").removeprefix("INT"))
        lower = 0 if datatype.startswith("UINT") else -(2 ** (bits - 1))
        upper = 2**bits - 1 if datatype.startswith("UINT") else 2 ** (bits - 1) - 1
        if np.any(values < lower) or np.any(values > upper):
            message = (
                "Integer model inputs are out of range"
                if is_request
                else "Integer model outputs are out of range"
            )
            (_invalid_request(message) if is_request else _invalid_response(message))


def _is_boolean_scalar(value: object) -> bool:
    """Identify Python and NumPy booleans before numeric coercion."""
    return isinstance(value, (bool, np.bool_))


def _reject_boolean_values(*, direction: _Direction) -> NoReturn:
    """Raise the direction-specific error for a non-BOOL boolean value."""
    message = (
        "Boolean model inputs are only valid for BOOL tensors"
        if direction == "request"
        else "Boolean model outputs are only valid for BOOL tensors"
    )
    if direction == "request":
        _invalid_request(message)
    _invalid_response(message)


def _validate_preconversion_values(
    values: object, datatype: str, *, direction: _Direction
) -> None:
    """Bound request values and reject booleans before NumPy conversion."""
    pending: list[object] = [values]
    count = 0
    while pending:
        current = pending.pop()
        if isinstance(current, np.ndarray):
            count += int(current.size)
            if direction == "request" and count > _MAX_REQUEST_ELEMENTS:
                _invalid_request("Model input request is too large")
            if datatype != "BOOL" and current.dtype.kind == "b":
                _reject_boolean_values(direction=direction)
            continue
        if isinstance(current, (list, tuple)):
            pending.extend(current)
            continue
        count += 1
        if direction == "request" and count > _MAX_REQUEST_ELEMENTS:
            _invalid_request("Model input request is too large")
        if datatype != "BOOL" and _is_boolean_scalar(current):
            _reject_boolean_values(direction=direction)


def _numeric_matrix(
    values: object, datatype: str, *, direction: _Direction
) -> np.ndarray:
    """Convert a matrix boundary while requiring finite supported numeric values."""
    _validate_preconversion_values(values, datatype, direction=direction)
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        if direction == "request":
            message = "Model inputs must be a finite numeric matrix"
            raise ProviderInvalidRequestError(message) from exc
        message = "model output must be numeric"
        raise ProviderInvalidResponseError(message) from exc
    if array.dtype.kind not in "bfiu":
        if direction == "request":
            _invalid_request("Model inputs must be a finite numeric matrix")
        _invalid_response("model output must be numeric")
    if not np.isfinite(array).all():
        if direction == "request":
            _invalid_request("Model inputs must be a finite numeric matrix")
        _invalid_response("model output must be finite")
    return array


def _input_row_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the per-row shape represented by negotiated input metadata."""
    return () if len(shape) == 1 and shape[0] == -1 else shape[-1:]


def _wire_values(values: np.ndarray, datatype: str) -> list[object]:
    """Convert validated NumPy scalars to JSON-native values for the wire type."""
    flattened = values.reshape(-1)
    if datatype == "BOOL":
        return [bool(value) for value in flattened]
    if datatype.startswith(("INT", "UINT")):
        return [int(value) for value in flattened]
    return [float(value) for value in flattened]


def encode_request(
    metadata: PredictionMetadata,
    inputs: np.ndarray,
    output_name: str | None = None,
) -> dict[str, object]:
    """Encode a finite numeric matrix as one bounded KServe V2 request."""
    _validate_numeric_datatype(metadata.input_datatype, direction="request")
    _validate_metadata_shape(metadata.input_shape)
    if output_name is not None and output_name != metadata.output_name:
        _invalid_request("Requested output selector does not match model metadata")
    values = _numeric_matrix(inputs, metadata.input_datatype, direction="request")
    if values.ndim != _MATRIX_RANK:
        _invalid_request("Model inputs must be a finite numeric matrix")
    if values.size > _MAX_REQUEST_ELEMENTS:
        _invalid_request("Model input request is too large")
    _validate_numeric_values(values, metadata.input_datatype, direction="request")

    row_shape = _input_row_shape(metadata.input_shape)
    expected_width = 1 if not row_shape else row_shape[0]
    if values.shape[1] != expected_width:
        if metadata.input_shape == (-1,):
            _invalid_request("Dynamic scalar input metadata requires one feature")
        _invalid_request("Input feature width does not match model metadata")
    if (
        len(metadata.input_shape) == _MATRIX_RANK
        and metadata.input_shape[0] == 1
        and values.shape[0] != 1
    ):
        _invalid_request("Fixed model batch size requires one input row")

    shape = [len(values), *row_shape] if row_shape else [len(values)]
    body: dict[str, object] = {
        "inputs": [
            {
                "name": metadata.input_name,
                "shape": shape,
                "datatype": metadata.input_datatype,
                "data": _wire_values(values, metadata.input_datatype),
            }
        ]
    }
    body["outputs"] = [{"name": metadata.output_name}]
    return body


def _validate_status(response: BoundedResponse) -> None:
    """Map HTTP status failures at the provider boundary."""
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, bool) or not isinstance(status_code, Integral):
        _invalid_response("model response has an invalid status")
    status = int(status_code)
    if status in {401, 403, 404, 408, 429} or status >= _HTTP_SERVER_ERROR:
        raise ProviderUnavailableError
    if status < _HTTP_SUCCESS_MIN or status >= _HTTP_REDIRECT:
        _invalid_response("Model inference request was rejected")


def _check_response_size(response: BoundedResponse) -> None:
    """Reject a materialized body before invoking its JSON parser."""
    try:
        content = getattr(response, "content", None)
        if content is not None and len(content) > _MAX_RESPONSE_BYTES:
            _invalid_response("model response is too large")
    except ProviderInvalidResponseError:
        raise
    except (AttributeError, TypeError) as exc:
        message = "model response has invalid content"
        raise ProviderInvalidResponseError(message) from exc


def _count_json_elements(value: object) -> int:
    """Count JSON leaves with a hard stop at the response element cap."""
    if not isinstance(value, (list, tuple)):
        _invalid_response("model output data must be a JSON array")
    count = 0
    pending: list[object] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, (list, tuple)):
            pending.extend(current)
            continue
        count += 1
        if count > _MAX_RESPONSE_ELEMENTS:
            _invalid_response("model output is too large")
    return count


def _parse_response_shape(value: object) -> tuple[int, ...]:
    """Parse an actual response shape, which cannot contain dynamic dimensions."""
    shape = _parse_shape(value)
    if (
        not shape
        or len(shape) > _MATRIX_RANK
        or any(dimension <= 0 for dimension in shape)
    ):
        _invalid_response("output shape mismatch")
    return shape


def _bounded_product(shape: tuple[int, ...]) -> int:
    """Multiply shape dimensions without allowing an oversized intermediate."""
    result = 1
    for dimension in shape:
        if result > _MAX_RESPONSE_ELEMENTS // dimension:
            _invalid_response("model output is too large")
        result *= dimension
    return result


def _allowed_output_shapes(
    metadata_shape: tuple[int, ...], chunk_size: int
) -> tuple[tuple[int, ...], ...]:
    """Return declared response shapes accepted for selected metadata."""
    if (
        len(metadata_shape) == _MATRIX_RANK
        and metadata_shape[0] == 1
        and chunk_size != 1
    ):
        _invalid_response("Fixed model batch size requires one response row")
    width = (
        1
        if len(metadata_shape) == 1 and metadata_shape[0] == -1
        else (metadata_shape[0] if len(metadata_shape) == 1 else metadata_shape[1])
    )
    if width == 1:
        return ((chunk_size,), (chunk_size, 1))
    return ((chunk_size, width),)


def _decode_output_tensor(
    selected: Mapping[str, object],
    metadata: PredictionMetadata,
    chunk_size: int,
) -> np.ndarray:
    """Validate and normalize one selected output tensor."""
    _validate_numeric_datatype(metadata.output_datatype, direction="response")
    _validate_metadata_shape(metadata.output_shape)
    if selected.get("datatype") != metadata.output_datatype:
        _invalid_response("response datatype mismatch")
    actual_shape = _parse_response_shape(selected.get("shape"))
    if actual_shape not in _allowed_output_shapes(metadata.output_shape, chunk_size):
        _invalid_response("output shape mismatch")
    raw_data = selected.get("data")
    element_count = _count_json_elements(raw_data)
    expected_count = _bounded_product(actual_shape)
    if element_count != expected_count:
        _invalid_response("output shape mismatch")
    _validate_preconversion_values(
        raw_data, metadata.output_datatype, direction="response"
    )

    try:
        data = np.asarray(raw_data)
    except (TypeError, ValueError) as exc:
        message = "model output must be numeric"
        raise ProviderInvalidResponseError(message) from exc
    if data.size > _MAX_RESPONSE_ELEMENTS:
        _invalid_response("model output is too large")
    if data.dtype.kind not in "bfiu":
        _invalid_response("model output must be numeric")
    if data.ndim != 1 and tuple(data.shape) != actual_shape:
        _invalid_response("output shape mismatch")
    _validate_numeric_values(data, metadata.output_datatype, direction="response")
    normalized = data.astype(float, copy=False)
    if not np.isfinite(normalized).all():
        _invalid_response("model output must be finite")

    width = (
        1
        if len(metadata.output_shape) == 1 and metadata.output_shape[0] == -1
        else (
            metadata.output_shape[0]
            if len(metadata.output_shape) == 1
            else metadata.output_shape[1]
        )
    )
    return (
        normalized.reshape(chunk_size, 1)
        if width == 1
        else normalized.reshape(chunk_size, width)
    )


def decode_response(
    response: BoundedResponse,
    metadata: PredictionMetadata,
    model_name: str,
    model_version: str | None,
    chunk_size: int,
) -> np.ndarray:
    """Decode one bounded KServe V2 response into a normalized NumPy matrix."""
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        _invalid_response("response batch size is invalid")
    try:
        _validate_status(response)
        _check_response_size(response)
        payload = response.json()
        if not isinstance(payload, Mapping):
            _invalid_response("model inference response must be a JSON object")
        if payload.get("model_name") != model_name:
            _invalid_response("response model name mismatch")
        if (
            model_version is not None
            and "model_version" in payload
            and payload["model_version"] != model_version
        ):
            _invalid_response("response model version mismatch")
        outputs = _tensor_list(payload, "outputs", allow_missing=False)
        selected_outputs = [
            output for output in outputs if output.get("name") == metadata.output_name
        ]
        if len(selected_outputs) != 1:
            _invalid_response("selected output is missing or duplicated")
        return _decode_output_tensor(selected_outputs[0], metadata, chunk_size)
    except (
        ProviderInvalidRequestError,
        ProviderInvalidResponseError,
        ProviderUnavailableError,
        ProviderUnsupportedModelError,
    ):
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ) as exc:
        raise ProviderInvalidResponseError from exc


# Descriptive aliases keep the codec boundary easy to discover for the provider.
metadata_from_payload = parse_metadata
encode_inference_request = encode_request
decode_inference_response = decode_response
