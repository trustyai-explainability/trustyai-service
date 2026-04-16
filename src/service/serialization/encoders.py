"""
JSON encoders and decoders for custom types.

Provides encoding/decoding for types that aren't natively JSON-serializable:
- NumPy arrays and scalars
- Binary data (via base64)
"""

import base64

import numpy as np


def json_encoder(obj):
    """
    Custom JSON encoder for numpy arrays, scalars, and binary data.

    Args:
        obj: Object to encode

    Returns:
        JSON-serializable representation

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
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, bytes):
        # Use base64 encoding for binary data (JSON can't handle arbitrary bytes)
        return {"__type__": "bytes", "data": base64.b64encode(obj).decode("ascii")}
    return str(obj)


def json_decoder_hook(obj):
    """
    Custom JSON decoder hook for special types.

    Decodes objects that were encoded with special markers by json_encoder.

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
            raise ValueError("Bytes object missing 'data' field")
        return base64.b64decode(obj["data"])
    return obj
