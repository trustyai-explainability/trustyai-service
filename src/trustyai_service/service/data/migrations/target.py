"""Database target adapter for the canonical migration contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from sqlalchemy.ext.asyncio import AsyncEngine

from trustyai_service.service.data.storage.db import MigrationStateStore

from .canonical import CanonicalStorage
from .contracts import DatasetDescriptor, MigrationTarget, PartialPayloadRecord


class DatabaseTarget(MigrationTarget):
    """Write canonical migration data through dialect-neutral SQLAlchemy."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        migration_namespace: str = "default",
    ) -> None:
        """Initialize canonical service storage and namespaced checkpoints."""
        self.engine = engine
        self.migration_namespace = migration_namespace
        self.storage = CanonicalStorage(engine)
        self.checkpoint_storage = MigrationStateStore(
            engine, namespace=migration_namespace
        )

    async def initialize(self) -> None:
        """Create target tables lazily."""
        await self.storage.initialize()
        await self.checkpoint_storage.initialize()

    async def ensure_dataset(self, dataset: DatasetDescriptor) -> None:
        """Create or validate a logical dataset registry row."""
        await self.storage.ensure_dataset(
            dataset.name,
            dataset.column_names,
            aliases=dataset.aliases,
            column_types=dataset.column_types,
            shape=dataset.shape or (len(dataset.column_names),),
            serialized=dataset.serialized,
            service_metadata=dataset.service_metadata or None,
        )

    async def write_batch(self, dataset: DatasetDescriptor, rows: np.ndarray) -> None:
        """Append one batch to the destination dataset."""
        await self.storage.append_data(dataset.name, rows, dataset.column_names)

    async def rollback_dataset(self, dataset_name: str) -> None:
        """Clear rows while retaining the canonical dataset definition."""
        await self.storage.clear_dataset_rows(dataset_name)

    async def persist_partial(self, payload: PartialPayloadRecord) -> None:
        """Persist a canonical partial payload."""
        await self.storage.persist_partial_payload(
            payload.payload,
            payload.payload_id,
            is_input=payload.is_input,
        )

    async def get_migration_run(self, migration_id: str) -> dict[str, Any] | None:
        """Read target-side migration state."""
        return await self.checkpoint_storage.get_migration_run(migration_id)

    async def save_migration_run(self, values: dict[str, Any]) -> None:
        """Write target-side migration state."""
        await self.checkpoint_storage.save_migration_run(values)

    async def get_migration_dataset_state(
        self, migration_id: str, dataset_name: str
    ) -> dict[str, Any] | None:
        """Read target-side dataset state."""
        return await self.checkpoint_storage.get_migration_dataset_state(
            migration_id,
            dataset_name,
        )

    async def save_migration_dataset_state(self, values: dict[str, Any]) -> None:
        """Write target-side dataset state."""
        await self.checkpoint_storage.save_migration_dataset_state(values)

    async def list_target_datasets(self) -> list[str]:
        """List service datasets for fresh-target protection."""
        return await self.storage.list_all_datasets()
