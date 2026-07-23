"""Data type enumeration for schema field types."""

from enum import StrEnum


class DataType(StrEnum):
    """Enumeration of supported data types."""

    BOOL = "BOOL"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    INT32 = "INT32"
    INT64 = "INT64"
    STRING = "STRING"
    TENSOR = "TENSOR"
    MAP = "MAP"
    UNKNOWN = "UNKNOWN"
