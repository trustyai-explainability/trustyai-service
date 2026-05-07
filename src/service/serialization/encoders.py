"""JSON encoders and decoders for custom types.

Provides encoding/decoding for types that aren't natively JSON-serializable:
- NumPy arrays and scalars
- Binary data (via base64)
"""

from __future__ import annotations

import base64

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
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, bytes):
        # Use base64 encoding for binary data (JSON can't handle arbitrary bytes)
        return {"__type__": "bytes", "data": base64.b64encode(obj).decode("ascii")}
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


def json_decoder_hook(obj: object) -> object:
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
    if isinstance(obj, dict) and obj.get("__type__") == "bytes":
        if "data" not in obj:
            msg = "Bytes object missing 'data' field"
            raise ValueError(msg)
        return base64.b64decode(obj["data"])
    return obj
