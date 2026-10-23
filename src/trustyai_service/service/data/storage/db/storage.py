"""Durable, namespaced migration checkpoints, separate from service data."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, select

from .schema import build_tables, create_schema

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class MigrationStateStore:
    """Persist migration runs and per-dataset outcomes only."""

    def __init__(self, engine: AsyncEngine, *, namespace: str = "default") -> None:
        """Bind tracking tables to one database and migration namespace."""
        self.engine = engine
        self.tables = build_tables(namespace)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create only the two checkpoint tables, once per store instance."""
        async with self._init_lock:
            if not self._initialized:
                await create_schema(self.tables.metadata, self.engine)
                self._initialized = True

    async def get_migration_run(self, migration_id: str) -> dict[str, Any] | None:
        """Read the durable outcome of a migration."""
        await self.initialize()
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(self.tables.runs).where(
                    self.tables.runs.c.migration_id == migration_id
                )
            )
            row = result.mappings().first()
            return dict(row) if row is not None else None

    async def save_migration_run(self, values: dict[str, Any]) -> None:
        """Insert a run or update its outcome without changing its identity."""
        await self.initialize()
        table = self.tables.runs
        async with self.engine.begin() as connection:
            where = table.c.migration_id == values["migration_id"]
            existing = await connection.execute(
                select(table.c.migration_id).where(where)
            )
            if existing.first() is None:
                await connection.execute(table.insert().values(**values))
            else:
                await connection.execute(
                    table.update()
                    .where(where)
                    .values(
                        **{
                            key: value
                            for key, value in values.items()
                            if key != "migration_id"
                        }
                    )
                )

    async def get_migration_dataset_state(
        self, migration_id: str, dataset_name: str
    ) -> dict[str, Any] | None:
        """Read the last transfer outcome for a dataset."""
        await self.initialize()
        table = self.tables.dataset_states
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(table).where(
                    (table.c.migration_id == migration_id)
                    & (table.c.dataset_name == dataset_name)
                )
            )
            row = result.mappings().first()
            return dict(row) if row is not None else None

    async def save_migration_dataset_state(self, values: dict[str, Any]) -> None:
        """Insert or update a dataset checkpoint."""
        await self.initialize()
        table = self.tables.dataset_states
        async with self.engine.begin() as connection:
            where = (table.c.migration_id == values["migration_id"]) & (
                table.c.dataset_name == values["dataset_name"]
            )
            existing = await connection.execute(select(table.c.status).where(where))
            if existing.first() is None:
                await connection.execute(table.insert().values(**values))
            else:
                await connection.execute(
                    table.update()
                    .where(where)
                    .values(
                        **{
                            key: value
                            for key, value in values.items()
                            if key not in {"migration_id", "dataset_name"}
                        }
                    )
                )

    async def reset_for_test(self) -> None:
        """Clear this namespace's checkpoints without touching canonical data."""
        await self.initialize()
        async with self.engine.begin() as connection:
            await connection.execute(delete(self.tables.dataset_states))
            await connection.execute(delete(self.tables.runs))
