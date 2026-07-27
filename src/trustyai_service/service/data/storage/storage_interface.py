"""Abstract storage interface defining the contract for data persistence backends."""

from abc import ABC, abstractmethod

import numpy as np

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.service.data.metadata.storage_metadata import StorageMetadata
from trustyai_service.service.data.modelmesh_parser import PartialPayload


class StorageInterface(ABC):
    """Abstract interface for data storage backends."""

    @abstractmethod
    async def dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset exists in storage."""

    @abstractmethod
    async def list_all_datasets(self) -> list[str]:
        """List all datasets stored in the backend."""

    @abstractmethod
    async def dataset_rows(self, dataset_name: str) -> int:
        """Get the number of rows in a dataset."""

    @abstractmethod
    async def dataset_shape(self, dataset_name: str) -> tuple[int]:
        """Get the shape of a dataset."""

    @abstractmethod
    async def write_data(
        self, dataset_name: str, new_rows: np.ndarray, column_names: list[str]
    ) -> None:
        """Write new rows to a dataset."""

    @abstractmethod
    async def read_data(
        self, dataset_name: str, start_row: int | None = None, n_rows: int | None = None
    ) -> np.ndarray:
        """Read data from a dataset with optional row range."""

    @abstractmethod
    async def get_original_column_names(self, dataset_name: str) -> list[str]:
        """Get the original column names from the raw payloads."""

    @abstractmethod
    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        """Get the current aliased column names after name mapping."""

    @abstractmethod
    async def apply_name_mapping(
        self, dataset_name: str, name_mapping: dict[str, str]
    ) -> None:
        """Apply column name aliases to a dataset."""

    @abstractmethod
    async def clear_name_mapping(self, dataset_name: str) -> None:
        """Clear all column name aliases from a dataset."""

    @abstractmethod
    async def get_known_models(self) -> list[str]:
        """Get a list of all model IDs that have inference data stored."""

    @abstractmethod
    async def get_metadata(self, model_id: str) -> StorageMetadata | None:
        """Get metadata for a specific model including shapes, column names, etc.

        Returns None if metadata cannot be retrieved or model doesn't exist.
        """

    @abstractmethod
    async def delete_dataset(self, dataset_name: str) -> None:
        """Delete a dataset from storage."""

    @abstractmethod
    async def persist_partial_payload(
        self,
        payload: PartialPayload | KServeInferenceRequest | KServeInferenceResponse,
        payload_id: str,
        *,
        is_input: bool,
    ) -> None:
        """Persist a partial payload before reconciliation."""

    @abstractmethod
    async def get_partial_payload(
        self, payload_id: str, *, is_input: bool, is_modelmesh: bool
    ) -> PartialPayload | KServeInferenceRequest | KServeInferenceResponse | None:
        """Retrieve a stored partial payload."""

    @abstractmethod
    async def delete_partial_payload(self, payload_id: str, *, is_input: bool) -> None:
        """Delete a stored partial payload.

        Args:
            payload_id: The unique identifier for the inference request
            is_input: Whether to delete an input payload (True) or output payload (False)

        """

    @abstractmethod
    async def persist_modelmesh_payload(
        self, payload: PartialPayload, request_id: str, *, is_input: bool
    ) -> None:
        """Store a ModelMesh partial payload (either input or output) for later reconciliation.

        Args:
            payload: The partial payload to store
            request_id: A unique identifier for this inference request
            is_input: Whether this is an input payload (True) or output payload (False)

        """

    @abstractmethod
    async def get_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> PartialPayload | None:
        """Retrieve a stored ModelMesh payload by request ID.

        Args:
            request_id: The unique identifier for the inference request
            is_input: Whether to retrieve an input payload (True) or output payload (False)

        Returns:
            The retrieved payload, or None if not found

        """

    @abstractmethod
    async def delete_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> None:
        """Delete a stored ModelMesh payload.

        Args:
            request_id: The unique identifier for the inference request
            is_input: Whether to delete an input payload (True) or output payload (False)

        """
