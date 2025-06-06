from enum import Enum


class DataType(str, Enum):
    """
    Enumeration of supported data types.
    """

    BOOL = "BOOL"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    INT32 = "INT32"
    INT64 = "INT64"
    STRING = "STRING"
    TENSOR = "TENSOR"
    MAP = "MAP"
    UNKNOWN = "UNKNOWN"
