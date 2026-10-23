"""Canonical logical contracts shared by migration sources and targets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.serialization import deserialize_model

if TYPE_CHECKING:
    import numpy as np
    from pydantic import BaseModel

    from trustyai_service.endpoints.consumer import (
        KServeInferenceRequest,
        KServeInferenceResponse,
    )


@dataclass(frozen=True)
class DatasetDescriptor:
    """Metadata needed to create and verify one logical dataset."""

    name: str
    column_names: tuple[str, ...]
    aliases: tuple[str, ...]
    column_types: tuple[str, ...]
    row_count: int
    shape: tuple[int, ...] = ()
    serialized: bool = False
    source_file: str | None = None
    service_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PartialPayloadRecord:
    """One in-flight KServe or ModelMesh payload."""

    payload_id: str
    is_input: bool
    is_modelmesh: bool
    payload: Any


@dataclass(frozen=True)
class SourceManifest:
    """Stable source snapshot metadata used for resume and verification."""

    source_type: str
    source_identity: str
    digest: str
    total_files: int
    datasets: tuple[DatasetDescriptor, ...]
    partial_payloads: tuple[PartialPayloadRecord, ...] = field(default_factory=tuple)

    @property
    def total_rows(self) -> int:
        """Return the number of normal dataset rows in the snapshot."""
        return sum(dataset.row_count for dataset in self.datasets)


def partial_payload_fingerprint(payload: PartialPayloadRecord) -> dict[str, Any]:
    """Return the JSON-compatible content used in a source digest."""
    value = payload.payload
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")
    return {
        "payload_id": payload.payload_id,
        "is_input": payload.is_input,
        "is_modelmesh": payload.is_modelmesh,
        "payload": value,
    }


def decode_partial_payload(
    raw: bytes, kserve_class: type[KServeInferenceRequest | KServeInferenceResponse]
) -> tuple[BaseModel, bool]:
    """Decode one stored partial payload and report whether it is ModelMesh.

    Backends persist KServe and ModelMesh partial payloads side by side without
    a type marker, so classification relies on the schemas being mutually
    exclusive: a serialized ``PartialPayload`` never validates as a KServe
    request/response and vice versa.
    """
    try:
        return deserialize_model(raw, kserve_class), False
    except ValueError:
        return deserialize_model(raw, PartialPayload), True


class MigrationSource(ABC):
    """Explicit source contract used by :class:`MigrationRunner`."""

    source_type: str
    quiesced: bool

    @abstractmethod
    async def preflight(self) -> SourceManifest:
        """Validate the source and return a snapshot manifest."""
        raise NotImplementedError

    @abstractmethod
    async def read_rows(
        self, dataset: DatasetDescriptor, start_row: int, n_rows: int
    ) -> np.ndarray:
        """Read one bounded logical row batch."""
        raise NotImplementedError


class VerificationStorage(Protocol):
    """The read-only destination view required for content verification."""

    async def dataset_exists(self, dataset_name: str) -> bool:
        """Return whether the dataset exists."""
        ...

    async def dataset_rows(self, dataset_name: str) -> int:
        """Return its stored row count."""
        ...

    async def get_original_column_names(self, dataset_name: str) -> list[str]:
        """Return original column names in order."""
        ...

    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        """Return effective aliases in order."""
        ...

    async def get_column_types(self, dataset_name: str) -> list[str] | None:
        """Return recorded logical column types."""
        ...

    async def get_service_metadata(self, dataset_name: str) -> dict[str, Any] | None:
        """Return runtime metadata needed by the service."""
        ...

    async def read_data(
        self, dataset_name: str, start_row: int | None = None, n_rows: int | None = None
    ) -> np.ndarray:
        """Read a bounded logical row range."""
        ...

    async def get_partial_payload(
        self, payload_id: str, *, is_input: bool, is_modelmesh: bool
    ) -> object:
        """Read an input or output partial payload."""
        ...


class MigrationTarget(ABC):
    """Explicit target contract used by :class:`MigrationRunner`."""

    migration_namespace: str
    storage: VerificationStorage

    @abstractmethod
    async def list_target_datasets(self) -> list[str]:
        """List all service datasets, excluding migration bookkeeping."""
        raise NotImplementedError

    @abstractmethod
    async def initialize(self) -> None:
        """Prepare target tables."""
        raise NotImplementedError

    @abstractmethod
    async def ensure_dataset(self, dataset: DatasetDescriptor) -> None:
        """Create or validate target dataset metadata."""
        raise NotImplementedError

    @abstractmethod
    async def write_batch(self, dataset: DatasetDescriptor, rows: np.ndarray) -> None:
        """Append one bounded row batch."""
        raise NotImplementedError

    @abstractmethod
    async def rollback_dataset(self, dataset_name: str) -> None:
        """Remove a dataset after a failed dataset-level transfer."""
        raise NotImplementedError

    @abstractmethod
    async def persist_partial(self, payload: PartialPayloadRecord) -> None:
        """Persist one partial payload."""
        raise NotImplementedError

    @abstractmethod
    async def get_migration_run(self, migration_id: str) -> dict[str, Any] | None:
        """Read durable migration state."""
        raise NotImplementedError

    @abstractmethod
    async def save_migration_run(self, values: dict[str, Any]) -> None:
        """Write durable migration state."""
        raise NotImplementedError

    @abstractmethod
    async def get_migration_dataset_state(
        self, migration_id: str, dataset_name: str
    ) -> dict[str, Any] | None:
        """Read durable dataset state."""
        raise NotImplementedError

    @abstractmethod
    async def save_migration_dataset_state(self, values: dict[str, Any]) -> None:
        """Write durable dataset state."""
        raise NotImplementedError
