"""JSON encoders and decoders for custom types.

Provides encoding/decoding for types that aren't natively JSON-serializable:
- NumPy arrays and scalars
- Binary data (via base64)
- Datetime objects (via ISO 8601)
"""

from __future__ import annotations

import base64
import datetime

import numpy as np


def json_encoder(obj: object) -> object:
    r"""Encode numpy arrays, scalars, and binary data for JSON serialization.

    Args:
        obj: Object to encode

    Returns:
        JSON-serializable representation

    Raises:
        TypeError: If object type is not supported

    Examples:
        >>> json_encoder(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> json_encoder(np.int64(42))
        42
        >>> json_encoder(b'\\x00\\x01\\xff')
        {"__type__": "bytes", "data": "AAEB/w=="}

    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, bytes):
        # Use base64 encoding for binary data (JSON can't handle arbitrary bytes)
        return {"__type__": "bytes", "data": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, datetime.datetime):
        return {"__type__": "datetime", "data": obj.isoformat()}
    if isinstance(obj, datetime.date):
        return {"__type__": "date", "data": obj.isoformat()}
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


def json_decoder_hook(obj: object) -> object:  # noqa: C901 -- validation requires checking multiple __type__ cases
    r"""Decode objects that were encoded with special markers by json_encoder.

    Args:
        obj: Object from JSON deserialization

    Returns:
        Decoded object (e.g., bytes from base64)

    Examples:
        >>> json_decoder_hook({"__type__": "bytes", "data": "AAEB/w=="})
        b'\\x00\\x01\\xff'
        >>> json_decoder_hook({"normal": "dict"})
        {"normal": "dict"}

    """
    if isinstance(obj, dict) and "__type__" in obj:
        type_tag = obj["__type__"]
        if type_tag == "bytes":
            if "data" not in obj:
                msg = "Bytes object missing 'data' field"
                raise ValueError(msg)
            try:
                return base64.b64decode(obj["data"], validate=True)
            except Exception as e:
                msg = f"Invalid base64 data in bytes object: {e}"
                raise ValueError(msg) from e
        if type_tag == "datetime":
            if "data" not in obj:
                msg = "Datetime object missing 'data' field"
                raise ValueError(msg)
            try:
                return datetime.datetime.fromisoformat(obj["data"])
            except ValueError as e:
                msg = f"Invalid ISO datetime format: {e}"
                raise ValueError(msg) from e
        if type_tag == "date":
            if "data" not in obj:
                msg = "Date object missing 'data' field"
                raise ValueError(msg)
            try:
                return datetime.date.fromisoformat(obj["data"])
            except ValueError as e:
                msg = f"Invalid ISO date format: {e}"
                raise ValueError(msg) from e
    return obj
