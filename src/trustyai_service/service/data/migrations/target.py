"""Database target adapter for the canonical migration contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from sqlalchemy.ext.asyncio import AsyncEngine

from trustyai_service.service.data.storage.db import DBStorage

from .canonical import CanonicalStorage
from .contracts import DatasetDescriptor, MigrationTarget, PartialPayloadRecord
from .errors import MigrationWriteError


class DatabaseTarget(MigrationTarget):
    """Write canonical migration data through dialect-neutral SQLAlchemy."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        migration_namespace: str = "default",
        fail_after_batches: int | None = None,
    ) -> None:
        """Initialize canonical service storage and namespaced checkpoints."""
        self.engine = engine
        self.migration_namespace = migration_namespace
        self.storage = CanonicalStorage(engine)
        self.checkpoint_storage = DBStorage(engine, namespace=migration_namespace)
        self._fail_after_batches = fail_after_batches
        self._batches_written = 0

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
        )

    async def write_batch(self, dataset: DatasetDescriptor, rows: np.ndarray) -> None:
        """Append one batch, optionally injecting a deterministic test failure."""
        if (
            self._fail_after_batches is not None
            and self._batches_written >= self._fail_after_batches
        ):
            msg = "injected target batch failure"
            raise MigrationWriteError(msg)
        await self.storage.append_data(dataset.name, rows, dataset.column_names)
        self._batches_written += 1

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

    async def reset_for_test(self) -> None:
        """Clear canonical data and this target's checkpoints explicitly."""
        await self.storage.reset_for_test()
        await self.checkpoint_storage.reset_for_test()
        self._batches_written = 0

    async def list_target_datasets(self) -> list[str]:
        """List service datasets for fresh-target protection."""
        return await self.storage.list_all_datasets()

    def clear_failure(self) -> None:
        """Disable deterministic failure injection for resume tests."""
        self._fail_after_batches = None
        self._batches_written = 0
