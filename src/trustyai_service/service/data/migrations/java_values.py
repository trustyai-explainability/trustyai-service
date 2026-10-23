"""Validate parsed Java values without loading Java classes or executing code."""

from __future__ import annotations

import math
from typing import Any

from .errors import MigrationValidationError

TENSOR_CLASS = "org.kie.trustyai.explainability.model.tensor.Tensor"
_SCALAR_TYPES = {
    "java.lang.Boolean": bool,
    "java.lang.Byte": int,
    "java.lang.Double": float,
    "java.lang.Float": float,
    "java.lang.Integer": int,
    "java.lang.Long": int,
    "java.lang.Short": int,
    "java.lang.String": str,
}
_MAX_TENSOR_DIMENSIONS = 32
_JAVA_SERIALIZABLE_FLAG = 2


def validate_object(value: object, class_name: str, fields: set[str]) -> None:
    """Reject unexpected object-stream classes, fields and custom annotations."""
    descriptor = getattr(value, "classdesc", None)
    names = getattr(descriptor, "fields_names", ())
    if (
        getattr(descriptor, "name", None) != class_name
        or getattr(descriptor, "flags", None) != _JAVA_SERIALIZABLE_FLAG
        or len(names) != len(fields)
        or set(names) != fields
        or getattr(descriptor, "superclass", None) is not None
        or getattr(value, "annotations", None)
    ):
        msg = "Java serialized value has an unexpected class or structure"
        raise MigrationValidationError(msg)


def scalar_value(value: object, expected_class: str | None) -> object:
    """Accept only a matching boxed primitive or string, never a rich object."""
    actual_class = getattr(getattr(value, "classdesc", None), "name", None)
    if actual_class is None and isinstance(value, str):
        actual_class = "java.lang.String"
    if actual_class != expected_class or expected_class not in _SCALAR_TYPES:
        msg = "Java serialized value contains an unexpected element type"
        raise MigrationValidationError(msg)
    scalar = getattr(value, "value", value)
    expected_type = _SCALAR_TYPES[expected_class]
    if expected_type is bool:
        valid = isinstance(scalar, bool)
    elif expected_type is int:
        valid = isinstance(scalar, int) and not isinstance(scalar, bool)
    elif expected_type is float:
        valid = isinstance(scalar, (int, float)) and not isinstance(scalar, bool)
        if valid:
            scalar = float(scalar)
    else:
        valid = isinstance(scalar, str)
    if not valid:
        msg = f"Java serialized value contains an invalid {expected_class} value"
        raise MigrationValidationError(msg)
    return scalar


def tensor_value(tensor: Any) -> object:  # noqa: ANN401 -- javaobj exposes dynamically parsed object fields
    """Validate a Tensor's exact structure and convert it to nested lists."""
    validate_object(
        tensor,
        TENSOR_CLASS,
        {"dimension", "nEntries", "data", "datatype", "dimensions"},
    )
    try:
        dimensions = [int(value) for value in tensor.dimensions]
        flat_data = list(tensor.data)
        datatype = tensor.datatype.name
        entries = int(tensor.nEntries)
        dimension = int(tensor.dimension)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        msg = "Java serialized Tensor has invalid fields"
        raise MigrationValidationError(msg) from error
    if (
        not dimensions
        or len(dimensions) > _MAX_TENSOR_DIMENSIONS
        or any(size <= 0 for size in dimensions)
        or dimension != len(dimensions)
        or math.prod(dimensions) != entries
        or len(flat_data) != entries
        or datatype not in _SCALAR_TYPES
    ):
        msg = "Java serialized Tensor has inconsistent dimensions or element count"
        raise MigrationValidationError(msg)
    return _nest_tensor(
        [scalar_value(item, datatype) for item in flat_data], dimensions
    )


def _nest_tensor(values: list[object], shape: list[int]) -> object:
    """Reshape row-major values into the validated Tensor dimensions."""
    if len(shape) == 1:
        return values
    stride = math.prod(shape[1:])
    return [
        _nest_tensor(values[index : index + stride], shape[1:])
        for index in range(0, len(values), stride)
    ]


def legacy_value(wrapper: Any) -> object:  # noqa: ANN401 -- javaobj exposes dynamically parsed object fields
    """Decode the legacy SerializableObject wrapper through a strict allowlist."""
    validate_object(
        wrapper, "org.kie.trustyai.explainability.model.SerializableObject", {"object"}
    )
    value = wrapper.object
    if value is None:
        return None
    class_name = getattr(getattr(value, "classdesc", None), "name", None)
    if class_name == TENSOR_CLASS:
        return tensor_value(value)
    if class_name is None and isinstance(value, str):
        class_name = "java.lang.String"
    return scalar_value(value, class_name)
