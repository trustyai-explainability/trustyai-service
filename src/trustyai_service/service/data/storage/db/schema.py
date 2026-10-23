"""SQLAlchemy table definitions for namespaced migration storage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKeyConstraint,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.exc import DBAPIError

from trustyai_service.service.data.storage.sql.schema import (
    _is_concurrent_table_creation,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_MAX_NAMESPACE_LENGTH = 255
_SCHEMA_CREATE_ATTEMPTS = 3


@dataclass(frozen=True)
class MigrationTables:
    """Tables belonging to one migration namespace."""

    metadata: MetaData
    runs: Table
    dataset_states: Table


def _namespace_token(namespace: str) -> str:
    """Return a stable, identifier-safe namespace token."""
    if not namespace or len(namespace) > _MAX_NAMESPACE_LENGTH:
        msg = "migration namespace must be between 1 and 255 characters"
        raise ValueError(msg)
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]


async def create_schema(metadata: MetaData, engine: AsyncEngine) -> None:
    """Create namespace tables, tolerating another instance creating them first.

    ``checkfirst`` and ``CREATE`` are separate operations, so a competing
    process can create a table between them. Retry the complete existence
    check in a fresh transaction after a duplicate-object error, including on
    PostgreSQL where the failed DDL aborts the preceding transaction.
    """
    for attempt in range(_SCHEMA_CREATE_ATTEMPTS):
        try:
            async with engine.begin() as connection:
                await connection.run_sync(metadata.create_all)
        except DBAPIError as error:
            if (
                attempt == _SCHEMA_CREATE_ATTEMPTS - 1
                or not _is_concurrent_table_creation(error)
            ):
                raise
        else:
            return


def build_tables(namespace: str) -> MigrationTables:
    """Build isolated tables for a migration namespace.

    Names are derived from a digest rather than interpolating caller-provided
    text. This keeps identifiers bounded and safe on MariaDB and PostgreSQL.
    """
    prefix = f"trustyai_migration_{_namespace_token(namespace)}"
    metadata = MetaData()

    runs = Table(
        f"{prefix}_runs",
        metadata,
        Column("migration_id", String(64), primary_key=True),
        Column("source_type", String(64), nullable=False),
        Column("source_identity", String(128), nullable=False),
        Column("status", String(32), nullable=False),
        Column("total_files", BigInteger, nullable=False, default=0),
        Column("total_datasets", BigInteger, nullable=False, default=0),
        Column("total_rows", BigInteger, nullable=False, default=0),
        Column("error_message", Text, nullable=True),
    )
    dataset_states = Table(
        f"{prefix}_dataset_states",
        metadata,
        Column("migration_id", String(64), nullable=False),
        Column("dataset_name", String(255), nullable=False),
        Column("source_identity", String(128), nullable=False),
        Column("status", String(32), nullable=False),
        Column("rows_written", BigInteger, nullable=False, default=0),
        Column("error_message", Text, nullable=True),
        ForeignKeyConstraint(
            ["migration_id"],
            [f"{prefix}_runs.migration_id"],
            name=f"fk_{prefix}_dataset_state_run",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "migration_id",
            "dataset_name",
            name=f"uq_{prefix}_dataset_state",
        ),
    )
    return MigrationTables(
        metadata=metadata,
        runs=runs,
        dataset_states=dataset_states,
    )
