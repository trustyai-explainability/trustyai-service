from typing import Any, Dict

from src.service.payloads.values.data_type import DataType


class TypedValue:
    """
    A class that holds a value with its data type.
    """

    def __init__(self) -> None:
        self.type: DataType = None
        self.value: Dict[str, Any] = None

    def get_type(self) -> DataType:
        return self.type

    def set_type(self, type_value: DataType) -> None:
        self.type = type_value

    def get_value(self) -> Dict[str, Any]:
        return self.value

    def set_value(self, value: Dict[str, Any]) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"TypedValue(type={self.type}, value={self.value})"
