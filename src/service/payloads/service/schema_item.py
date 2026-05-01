"""Schema item definition representing a single field in a data schema."""

from dataclasses import dataclass

from src.service.payloads.values.data_type import DataType


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

    def set_type(
        self,
        data_type: DataType | None = None,
        *,
        type: DataType | None = None,  # noqa: A002  # Backward compatibility with old parameter name
    ) -> None:
        """Set the data type of this schema item.

        Args:
            data_type: New data type (preferred parameter name)
            type: Old parameter name for backward compatibility

        """
        # Support both parameter names for backward compatibility
        final_type = data_type if data_type is not None else type
        if final_type is None:
            msg = "Must provide either 'data_type' or 'type' parameter"
            raise TypeError(msg)
        self.type = final_type

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
