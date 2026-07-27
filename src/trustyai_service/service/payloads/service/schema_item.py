"""Schema item definition representing a single field in a data schema."""

from dataclasses import dataclass

from trustyai_service.service.payloads.values.data_type import DataType


@dataclass
class SchemaItem:
    """Represents a single field in a data schema with type and position."""

    type: DataType
    name: str
    column_index: int

    # Keeping these methods for backward compatibility
    def get_type(self) -> DataType:
        """Get the data type of this schema item."""
        return self.type

    def set_type(self, data_type: DataType) -> None:
        """Set the data type of this schema item."""
        self.type = data_type

    def get_name(self) -> str:
        """Get the name of this schema item."""
        return self.name

    def set_name(self, name: str) -> None:
        """Set the name of this schema item."""
        self.name = name

    def get_column_index(self) -> int:
        """Get the column index of this schema item."""
        return self.column_index

    def set_column_index(self, column_index: int) -> None:
        """Set the column index of this schema item."""
        self.column_index = column_index
