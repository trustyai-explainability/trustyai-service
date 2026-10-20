"""SQLAlchemy table definitions for namespaced migration storage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
from sqlalchemy.types import JSON

_MAX_NAMESPACE_LENGTH = 255
_SURROGATE_KEY_TYPE = BigInteger().with_variant(Integer, "sqlite")


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
        Column("metadata", JSON, nullable=False),
        Column("n_rows", BigInteger, nullable=False, default=0),
        UniqueConstraint("dataset_name", name=f"uq_{prefix}_dataset_name"),
    )
    data = Table(
        f"{prefix}_data",
        metadata,
        Column("id", _SURROGATE_KEY_TYPE, primary_key=True, autoincrement=True),
        Column("dataset_name", String(255), nullable=False),
        Column("row_idx", BigInteger, nullable=False),
        Column("row_data", LargeBinary, nullable=False),
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
        Column("payload_data", LargeBinary, nullable=False),
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
