"""Typed value container for data values with associated type information."""

from typing import Any

from trustyai_service.service.payloads.values.data_type import DataType


class TypedValue:
    """A class that holds a value with its data type."""

    def __init__(self) -> None:
        """Initialize an empty typed value container."""
        self.type: DataType | None = None
        self.value: dict[str, Any] | None = None

    def get_type(self) -> DataType | None:
        """Get the data type of this value."""
        return self.type

    def set_type(self, type_value: DataType) -> None:
        """Set the data type of this value."""
        self.type = type_value

    def get_value(self) -> dict[str, Any] | None:
        """Get the value data."""
        return self.value

    def set_value(self, value: dict[str, Any]) -> None:
        """Set the value data."""
        self.value = value

    def __str__(self) -> str:
        """Return string representation of the typed value."""
        return f"TypedValue(type={self.type}, value={self.value})"
