"""SQLAlchemy table definitions for namespaced migration storage."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    ForeignKeyConstraint,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.exc import DBAPIError
from sqlalchemy.types import JSON

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_MAX_NAMESPACE_LENGTH = 255
_SURROGATE_KEY_TYPE = BigInteger().with_variant(Integer, "sqlite")
_SCHEMA_CREATE_ATTEMPTS = 3
_MARIADB_TABLE_EXISTS = 1050
_POSTGRES_CATALOG_UNIQUE_CONSTRAINTS = {
    "pg_class_relname_nsp_index",
    "pg_type_typname_nsp_index",
}
# Opaque gzipped-JSON blobs. MySQL/MariaDB's default BLOB caps at 64 KiB,
# far too small for gzipped payload and row blobs, so LONGBLOB is used there.
_BLOB = (
    LargeBinary()
    .with_variant(mysql.LONGBLOB(), "mysql")
    .with_variant(mysql.LONGBLOB(), "mariadb")
)
# JSON metadata document. JSONB on PostgreSQL for parity with the SQL storage
# backends and indexability.
_JSON = JSON().with_variant(postgresql.JSONB(), "postgresql")


@dataclass(frozen=True)
class MigrationTables:
    """Tables belonging to one migration namespace."""

    metadata: MetaData
    datasets: Table
    data: Table
    partial_payloads: Table
    runs: Table
    dataset_states: Table


def _namespace_token(namespace: str) -> str:
    """Return a stable, identifier-safe namespace token."""
    if not namespace or len(namespace) > _MAX_NAMESPACE_LENGTH:
        msg = "migration namespace must be between 1 and 255 characters"
        raise ValueError(msg)
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]


def _is_concurrent_table_creation(error: DBAPIError) -> bool:
    """Recognize duplicate DDL objects without treating data conflicts as races."""
    if not (error.statement or "").lstrip().upper().startswith("CREATE TABLE "):
        return False
    original = error.orig
    if getattr(original, "errno", None) == _MARIADB_TABLE_EXISTS:
        return True
    sqlstate = getattr(original, "sqlstate", None)
    if sqlstate == "42P07":  # PostgreSQL duplicate_table
        return True
    if sqlstate == "23505":  # Concurrent CREATE can collide in system catalogs.
        diagnostic = getattr(original, "diag", None)
        return (
            getattr(diagnostic, "constraint_name", None)
            in _POSTGRES_CATALOG_UNIQUE_CONSTRAINTS
        )
    return (
        isinstance(original, sqlite3.OperationalError)
        and getattr(original, "sqlite_errorcode", None) == sqlite3.SQLITE_ERROR
        and str(original).startswith("table ")
        and str(original).endswith(" already exists")
    )


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

    datasets = Table(
        f"{prefix}_datasets",
        metadata,
        Column("id", _SURROGATE_KEY_TYPE, primary_key=True, autoincrement=True),
        Column("dataset_name", String(255), nullable=False),
        Column("metadata", _JSON, nullable=False),
        Column("n_rows", BigInteger, nullable=False, default=0),
        UniqueConstraint("dataset_name", name=f"uq_{prefix}_dataset_name"),
    )
    data = Table(
        f"{prefix}_data",
        metadata,
        Column("id", _SURROGATE_KEY_TYPE, primary_key=True, autoincrement=True),
        Column("dataset_name", String(255), nullable=False),
        Column("row_idx", BigInteger, nullable=False),
        Column("row_data", _BLOB, nullable=False),
        UniqueConstraint(
            "dataset_name",
            "row_idx",
            name=f"uq_{prefix}_dataset_row",
        ),
    )
    partial_payloads = Table(
        f"{prefix}_partial_payloads",
        metadata,
        Column("payload_id", String(255), nullable=False),
        Column("is_input", Boolean, nullable=False),
        Column("is_modelmesh", Boolean, nullable=False, default=False),
        Column("payload_data", _BLOB, nullable=False),
        UniqueConstraint(
            "payload_id",
            "is_input",
            "is_modelmesh",
            name=f"uq_{prefix}_partial_payload",
        ),
    )
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
        datasets=datasets,
        data=data,
        partial_payloads=partial_payloads,
        runs=runs,
        dataset_states=dataset_states,
    )
