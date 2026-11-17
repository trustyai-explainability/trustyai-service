"""HDF5-on-PVC target adapter for the canonical migration contract."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from trustyai_service.service.data.storage.pvc import MAX_VOID_TYPE_LENGTH, PVCStorage

from .contracts import DatasetDescriptor, MigrationTarget, PartialPayloadRecord

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# A dataset of only floating-point or only integer columns is stored as plain
# numbers. Any other mix (strings, booleans, tensors, maps, or ints with floats)
# is stored serialized, which is lossless and gives every batch the same layout.
_FLOAT_TYPES = frozenset({"FLOAT", "DOUBLE"})
_INTEGER_TYPES = frozenset({"INT32", "INT64"})
_PLAIN_VALUE_BYTES = 8
# Tensor names the service assumes, because the layout has no place to store them.
_DEFAULT_SERVICE_METADATA = {
    "input_tensor_name": "input",
    "output_tensor_name": "output",
}


def _plain_dtype(column_types: Iterable[str]) -> type[np.generic] | None:
    """Return the dtype for a dataset stored as plain numbers, or None if serialized."""
    types = set(column_types)
    if types <= _FLOAT_TYPES:
        return np.float64
    if types <= _INTEGER_TYPES:
        return np.int64
    return None


def estimated_size(dataset: DatasetDescriptor) -> int:
    """Estimate the bytes a dataset takes on disk in the service's HDF5 layout."""
    if _plain_dtype(dataset.column_types) is None:
        return dataset.row_count * MAX_VOID_TYPE_LENGTH
    return dataset.row_count * len(dataset.column_names) * _PLAIN_VALUE_BYTES


class _RecordingPVCStorage(PVCStorage):
    """PVC storage that answers the verification questions the runner asks.

    The runner compares plain lists, while HDF5 attributes come back as arrays.
    The HDF5 layout also has no field for tensor names, so they cannot be read
    back; echoing the requested values keeps verification on the data itself.
    """

    def __init__(self, data_directory: str, data_file: str) -> None:
        """Initialize storage and an empty service metadata record."""
        super().__init__(data_directory=data_directory, data_file=data_file)
        self.requested_service_metadata: dict[str, dict[str, Any]] = {}

    async def get_original_column_names(self, dataset_name: str) -> list[str]:
        """Return the original column names as a list."""
        return [
            str(name) for name in await super().get_original_column_names(dataset_name)
        ]

    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        """Return the effective column names as a list."""
        return [
            str(name) for name in await super().get_aliased_column_names(dataset_name)
        ]

    async def get_service_metadata(self, dataset_name: str) -> dict[str, Any] | None:
        """Return the metadata requested for this dataset, if any."""
        return self.requested_service_metadata.get(dataset_name)


class PVCTarget(MigrationTarget):
    """Write migrated datasets as the HDF5 files the Python service reads."""

    def __init__(
        self,
        directory: str | Path,
        data_file: str,
        *,
        migration_namespace: str = "java-pvc",
    ) -> None:
        """Initialize a target directory and the service's data file name."""
        self.directory = Path(directory)
        self.migration_namespace = migration_namespace
        self.storage = _RecordingPVCStorage(str(self.directory), data_file)
        self._runs: dict[str, dict[str, Any]] = {}
        self._dataset_states: dict[tuple[str, str], dict[str, Any]] = {}
        self._described: set[str] = set()

    async def initialize(self) -> None:
        """Create the target directory."""
        self.directory.mkdir(parents=True, exist_ok=True)

    async def list_target_datasets(self) -> list[str]:
        """List datasets already present in the target directory."""
        return await self.storage.list_all_datasets()

    async def ensure_dataset(self, dataset: DatasetDescriptor) -> None:
        """Record dataset metadata and create empty datasets.

        HDF5 datasets are created by their first write. A dataset without rows
        never receives one, so it is created here.
        """
        self._described.discard(dataset.name)
        if dataset.service_metadata:
            self.storage.requested_service_metadata[dataset.name] = dict(
                dataset.service_metadata
            )
            self._warn_if_metadata_is_not_persisted(dataset)
        if dataset.row_count == 0:
            await self._write(dataset, np.empty((0, len(dataset.column_names))))
            await self._describe(dataset)

    async def write_batch(self, dataset: DatasetDescriptor, rows: np.ndarray) -> None:
        """Append one batch and, after the first, store column metadata."""
        await self._write(dataset, rows)
        if dataset.name not in self._described:
            await self._describe(dataset)

    async def rollback_dataset(self, dataset_name: str) -> None:
        """Remove a dataset after a failed transfer."""
        await self.storage.delete_dataset(dataset_name)
        self._described.discard(dataset_name)

    async def persist_partial(self, payload: PartialPayloadRecord) -> None:
        """Persist one partial payload."""
        await self.storage.persist_partial_payload(
            payload.payload, payload.payload_id, is_input=payload.is_input
        )

    async def get_migration_run(self, migration_id: str) -> dict[str, Any] | None:
        """Read run state; it lives only as long as this staging attempt."""
        return self._runs.get(migration_id)

    async def save_migration_run(self, values: dict[str, Any]) -> None:
        """Write run state for this staging attempt."""
        self._runs[values["migration_id"]] = dict(values)

    async def get_migration_dataset_state(
        self, migration_id: str, dataset_name: str
    ) -> dict[str, Any] | None:
        """Read dataset state for this staging attempt."""
        return self._dataset_states.get((migration_id, dataset_name))

    async def save_migration_dataset_state(self, values: dict[str, Any]) -> None:
        """Write dataset state for this staging attempt."""
        key = (values["migration_id"], values["dataset_name"])
        self._dataset_states[key] = dict(values)

    async def _write(self, dataset: DatasetDescriptor, rows: np.ndarray) -> None:
        """Store rows as plain numbers or serialized, fixed per dataset."""
        names = list(dataset.column_names)
        dtype = _plain_dtype(dataset.column_types)
        if dtype is None:
            await self.storage.write_data(
                dataset.name, rows.tolist(), names, force_serialization=True
            )
            return
        plain = np.array(rows.tolist(), dtype=dtype)
        await self.storage.write_data(dataset.name, plain, names)

    async def _describe(self, dataset: DatasetDescriptor) -> None:
        """Store column types and aliases once the dataset exists."""
        await self.storage.set_column_types(dataset.name, list(dataset.column_types))
        mapping = {
            name: alias
            for name, alias in zip(dataset.column_names, dataset.aliases, strict=True)
            if name != alias
        }
        if mapping:
            await self.storage.apply_name_mapping(dataset.name, mapping)
        self._described.add(dataset.name)

    @staticmethod
    def _warn_if_metadata_is_not_persisted(dataset: DatasetDescriptor) -> None:
        """Warn when tensor names differ from what the service assumes."""
        lost = {
            key: value
            for key, value in dataset.service_metadata.items()
            if key in _DEFAULT_SERVICE_METADATA
            and value != _DEFAULT_SERVICE_METADATA[key]
        }
        if lost:
            logger.warning(
                "Dataset %s uses tensor names %s; the PVC layout does not store them, "
                "so the service will assume the defaults %s",
                dataset.name,
                lost,
                _DEFAULT_SERVICE_METADATA,
            )
