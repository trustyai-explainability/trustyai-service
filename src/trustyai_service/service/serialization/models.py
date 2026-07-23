"""Pydantic model serialization with JSON + gzip compression.

Provides secure serialization for Pydantic models using JSON for data
representation and gzip for compression.

This replaces pickle serialization to eliminate CWE-502 vulnerabilities.
Part of: RHOAIENG-56132 (Replace pickle with JSON+gzip)
Epic: RHOAIENG-55574 (Security Remediation Program)
"""

import gzip
import json
import logging
import zlib

from pydantic import BaseModel, ValidationError

from .detection import detect_format, safe_gzip_decompress
from .encoders import json_decoder_hook, json_encoder

logger = logging.getLogger(__name__)


def serialize_model(obj: BaseModel | dict) -> bytes:
    r"""Serialize a Pydantic model or dictionary using JSON + gzip.

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
        msg = f"Cannot serialize type: {type(obj).__name__}"
        raise ValueError(msg)

    # Serialize to JSON then compress with gzip
    json_str = json.dumps(data, default=json_encoder)
    return gzip.compress(json_str.encode("utf-8"))


def deserialize_model[T: BaseModel](data: bytes, target_class: type[T]) -> T:  # noqa: C901 -- multi-format deserialization requires complexity
    """Deserialize and validate data against expected Pydantic schema.

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
            json_str = safe_gzip_decompress(data).decode("utf-8")
            obj_dict = json.loads(json_str, object_hook=json_decoder_hook)
            if not isinstance(obj_dict, dict):
                msg = f"Expected JSON object (dict), got {type(obj_dict).__name__}"
                raise TypeError(msg)
            return target_class(**obj_dict)
        except ValidationError:
            raise
        except (
            OSError,
            gzip.BadGzipFile,
            zlib.error,
            EOFError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
        ) as e:
            if format_type == "gzip":
                msg = f"Failed to deserialize gzip data: {e}"
                raise ValueError(msg) from e

    # Try uncompressed JSON
    if format_type == "json" or format_type is None:
        try:
            obj_dict = json.loads(data.decode("utf-8"), object_hook=json_decoder_hook)
            if not isinstance(obj_dict, dict):
                msg = f"Expected JSON object (dict), got {type(obj_dict).__name__}"
                raise TypeError(msg)
            return target_class(**obj_dict)
        except ValidationError:
            raise
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            if format_type == "json":
                msg = f"Failed to deserialize JSON data: {e}"
                raise ValueError(msg) from e

    # Pickle is intentionally unsupported (CWE-502: deserialization of untrusted data).
    msg = (
        f"Unsupported serialization format. Expected JSON or gzip-compressed JSON, "
        f"got format: {format_type or 'unknown'}"
    )
    raise ValueError(msg)
