"""
Pydantic model serialization with JSON + gzip compression.

Provides secure serialization for Pydantic models using JSON for data
representation and gzip for compression.
"""

import gzip
import json
import logging

from pydantic import BaseModel

from .detection import detect_format
from .encoders import json_decoder_hook, json_encoder

logger = logging.getLogger(__name__)


def serialize_model(obj: BaseModel | dict) -> bytes:
    """
    Serialize a Pydantic model or dictionary using JSON + gzip.

    Args:
        obj: Pydantic model instance or dictionary to serialize

    Returns:
        Compressed JSON bytes

    Raises:
        ValueError: If object type is not supported

    Examples:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> user = User(name="Alice", age=30)
        >>> data = serialize_model(user)
        >>> data.startswith(b'\\x1f\\x8b')  # gzip magic bytes
        True
    """
    if isinstance(obj, dict):
        data = obj
    elif hasattr(obj, "model_dump"):
        data = obj.model_dump()
    else:
        raise ValueError(f"Cannot serialize type: {type(obj).__name__}")

    # Serialize to JSON then compress with gzip
    json_str = json.dumps(data, default=json_encoder)
    return gzip.compress(json_str.encode("utf-8"))


def deserialize_model[T: BaseModel](data: bytes, target_class: type[T]) -> T:
    """
    Deserialize and validate data against expected Pydantic schema.

    Supports both gzip-compressed and uncompressed JSON formats.

    Args:
        data: Serialized bytes
        target_class: Pydantic model class for validation

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If data is invalid or schema validation fails

    Examples:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> serialized = serialize_model(User(name="Bob", age=25))
        >>> user = deserialize_model(serialized, User)
        >>> user.name
        'Bob'
    """
    try:
        format_type = detect_format(data)
    except ValueError:
        # If we can't detect format, try all methods
        format_type = None

    # Try gzip-compressed JSON first (production format)
    if format_type == "gzip" or format_type is None:
        try:
            json_str = gzip.decompress(data).decode("utf-8")
            obj_dict = json.loads(json_str, object_hook=json_decoder_hook)
            return target_class(**obj_dict)
        except (OSError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError, ValueError):
            if format_type == "gzip":
                raise  # If we detected gzip but it failed, don't try other formats

    # Try uncompressed JSON
    if format_type == "json" or format_type is None:
        try:
            obj_dict = json.loads(data.decode("utf-8"), object_hook=json_decoder_hook)
            return target_class(**obj_dict)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            if format_type == "json":
                raise ValueError(f"Failed to deserialize JSON data: {e}") from e

    raise ValueError(
        f"Unsupported serialization format. Expected JSON or gzip-compressed JSON, "
        f"got format: {format_type if format_type else 'unknown'}"
    )
