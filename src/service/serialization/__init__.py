"""
Secure serialization utilities with JSON + gzip compression.

Replaces pickle serialization to eliminate arbitrary code execution risks
while maintaining backward compatibility with existing data.

Public API:
    - serialize_model(obj) -> bytes
    - deserialize_model(data, target_class) -> Model
    - serialize_rows(lst, max_void_type_length) -> ndarray
    - deserialize_rows(serialized) -> ndarray

Examples:
    >>> from pydantic import BaseModel
    >>> class User(BaseModel):
    ...     name: str
    ...     age: int
    >>>
    >>> # Serialize Pydantic model
    >>> user = User(name="Alice", age=30)
    >>> data = serialize_model(user)
    >>>
    >>> # Deserialize back
    >>> loaded = deserialize_model(data, User)
    >>> loaded.name
    'Alice'
    >>>
    >>> # Serialize rows
    >>> rows = [[1, 2, 3], [4, 5, 6]]
    >>> serialized = serialize_rows(rows, max_void_type_length=1024)
    >>> deserialized = deserialize_rows(serialized)
"""

from .models import deserialize_model, serialize_model
from .rows import deserialize_rows, serialize_rows

__all__ = [
    "deserialize_model",
    "deserialize_rows",
    "serialize_model",
    "serialize_rows",
]
