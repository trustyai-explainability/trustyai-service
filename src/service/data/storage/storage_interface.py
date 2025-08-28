from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

from src.endpoints.consumer import KServeInferenceResponse, KServeInferenceRequest
from src.service.data.modelmesh_parser import PartialPayload


class StorageInterface(ABC):
    @abstractmethod
    async def dataset_exists(self, dataset_name: str) -> bool:
        pass

    @abstractmethod
    async def list_all_datasets(self) -> List[str]:
        pass

    @abstractmethod
    async def dataset_rows(self, dataset_name: str) -> int:
        pass

    @abstractmethod
    async def dataset_shape(self, dataset_name: str) -> tuple[int]:
        pass

    @abstractmethod
    async def write_data(self, dataset_name: str, new_rows, column_names: List[str]):
        pass

    @abstractmethod
    async def read_data(self, dataset_name: str, start_row: int = None, n_rows: int = None):
        pass

    @abstractmethod
    async def get_original_column_names(self, dataset_name: str) -> List[str]:
        pass

    @abstractmethod
    async def get_aliased_column_names(self, dataset_name: str) -> List[str]:
        pass

    @abstractmethod
    async def apply_name_mapping(self, dataset_name: str, name_mapping: Dict[str, str]):
        pass

    @abstractmethod
    async def delete_dataset(self, dataset_name: str):
        pass

    @abstractmethod
    async def persist_partial_payload(self,
                                      payload: Union[PartialPayload, KServeInferenceRequest, KServeInferenceResponse],
                                      payload_id, is_input: bool):
        pass

    @abstractmethod
    async def get_partial_payload(self, payload_id: str, is_input: bool, is_modelmesh: bool) -> Optional[
        Union[PartialPayload, KServeInferenceRequest, KServeInferenceResponse]]:
        pass


    @abstractmethod
    async def delete_partial_payload(self, payload_id: str, is_input: bool):
        """
        Delete a stored partial payload.

        Args:
            request_id: The unique identifier for the inference request
            is_input: Whether to delete an input payload (True) or output payload (False)
        """
        pass
