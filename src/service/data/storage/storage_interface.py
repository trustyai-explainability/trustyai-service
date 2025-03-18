from abc import ABC, abstractmethod
from typing import List, Dict


class StorageInterface(ABC):
    @abstractmethod
    def dataset_exists(self, dataset_name: str) -> bool:
        pass

    @abstractmethod
    def dataset_rows(self, dataset_name: str) -> int:
        pass

    @abstractmethod
    def dataset_shape(self, dataset_name: str) -> tuple[int]:
        pass

    @abstractmethod
    async def write_data(self, dataset_name: str, new_rows, column_names: List[str]):
        pass

    @abstractmethod
    def read_data(self, dataset_name: str, start_row: int = None, n_rows: int = None):
        pass

    @abstractmethod
    def get_original_column_names(self, dataset_name: str) -> List[str]:
        pass

    @abstractmethod
    def get_aliased_column_names(self, dataset_name: str) -> List[str]:
        pass


    @abstractmethod
    def apply_name_mapping(self, dataset_name: str, name_mapping: Dict[str, str]):
        pass

    @abstractmethod
    def delete_dataset(self, dataset_name: str):
        pass
