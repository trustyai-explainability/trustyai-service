from dataclasses import dataclass

from src.service.payloads.values.data_type import DataType


@dataclass
class SchemaItem:
    type: DataType
    name: str
    column_index: int

    # Keeping these methods for backward compatibility
    def get_type(self) -> DataType:
        return self.type

    def set_type(self, type: DataType) -> None:
        self.type = type

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str) -> None:
        self.name = name

    def get_column_index(self) -> int:
        return self.column_index

    def set_column_index(self, column_index: int) -> None:
        self.column_index = column_index
