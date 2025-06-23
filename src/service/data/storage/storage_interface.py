from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from src.service.data.modelmesh_parser import PartialPayload


class StorageInterface(ABC):
    @abstractmethod
    def dataset_exists(self, dataset_name: str) -> bool:
        pass

    @abstractmethod
    def list_all_datasets(self) -> List[str]:
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

    @abstractmethod
    async def persist_partial_payload(self, payload, is_input: bool):
        pass

    @abstractmethod
    async def get_partial_payload(self, payload_id: str, is_input: bool):
        pass


    @abstractmethod
    async def persist_modelmesh_payload(
        self, payload: PartialPayload, request_id: str, is_input: bool
    ):
        """
        Store a ModelMesh partial payload (either input or output) for later reconciliation.

        Args:
            payload: The partial payload to store
            request_id: A unique identifier for this inference request
            is_input: Whether this is an input payload (True) or output payload (False)
        """
        pass

    @abstractmethod
    async def get_modelmesh_payload(
        self, request_id: str, is_input: bool
    ) -> Optional[PartialPayload]:
        """
        Retrieve a stored ModelMesh payload by request ID.

        Args:
            request_id: The unique identifier for the inference request
            is_input: Whether to retrieve an input payload (True) or output payload (False)

        Returns:
            The retrieved payload, or None if not found
        """
        pass

    @abstractmethod
    async def delete_modelmesh_payload(self, request_id: str, is_input: bool):
        """
        Delete a stored ModelMesh payload.

        Args:
            request_id: The unique identifier for the inference request
            is_input: Whether to delete an input payload (True) or output payload (False)
        """
        pass
