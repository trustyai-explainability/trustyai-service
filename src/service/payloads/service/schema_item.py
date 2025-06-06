from src.service.payloads.values.data_type import DataType


class SchemaItem:
    def __init__(self, type: DataType, name: str, column_index: int) -> None:
        self.type: DataType = type
        self.name: str = name
        self.column_index: int = column_index

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SchemaItem):
            return False
        return (
            self.type == other.type
            and self.name == other.name
            and self.column_index == other.column_index
        )

    def __hash__(self) -> int:
        return hash((self.type, self.name, self.column_index))

    def __str__(self) -> str:
        return f"SchemaItem(type={self.type}, name='{self.name}', column_index={self.column_index})"
