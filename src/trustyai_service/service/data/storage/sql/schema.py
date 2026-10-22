"""Dynamic ``Table``/``MetaData`` builders and per-dialect type mapping.

The data model uses *dynamic per-dataset tables* whose column count is known
only at runtime, with every cell an opaque gzipped-JSON blob. These helpers
build the SQLAlchemy Core ``Table`` objects for that model, mapping the abstract
column types to dialect-appropriate SQLAlchemy types.

Type mapping rationale (preserves the layout written by the raw-SQL backends):

- **auto-increment primary key** -- ``BigInteger`` compiles to ``BIGSERIAL``
  (PostgreSQL) / ``BIGINT AUTO_INCREMENT`` (MariaDB). SQLite only makes an
  ``INTEGER PRIMARY KEY`` an alias for ``rowid`` (its autoincrement), so the key
  uses an ``Integer`` variant on SQLite.
- **blob cells / payloads** -- ``LargeBinary`` compiles to ``BYTEA`` (PostgreSQL)
  and ``BLOB`` (SQLite). On MySQL/MariaDB the default ``BLOB`` caps at 64 KiB,
  far too small for gzipped payloads, so a ``LONGBLOB`` variant is used there to
  match the raw-SQL MariaDB backend.
- **metadata document** -- ``JSON`` compiles to ``JSONB`` on PostgreSQL (matching
  the raw-SQL backend) and to ``JSON``/``TEXT`` elsewhere.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
)
from sqlalchemy.dialects import mysql, postgresql
from sqlalchemy.exc import DBAPIError

if TYPE_CHECKING:
    from sqlalchemy import Engine

_SCHEMA_CREATE_ATTEMPTS = 3
_MARIADB_TABLE_EXISTS = 1050
_POSTGRES_CATALOG_UNIQUE_CONSTRAINTS = {
    "pg_class_relname_nsp_index",
    "pg_type_typname_nsp_index",
}

# Auto-increment integer primary key, dialect-correct on every backend.
_AUTOINCREMENT_PK = BigInteger().with_variant(Integer, "sqlite")

# Opaque gzipped-JSON blob cell. LONGBLOB on MariaDB/MySQL to exceed the 64 KiB
# default BLOB ceiling.
_BLOB = (
    LargeBinary()
    .with_variant(mysql.LONGBLOB(), "mysql")
    .with_variant(mysql.LONGBLOB(), "mariadb")
)

# JSON metadata document. JSONB on PostgreSQL for indexability + parity.
_JSON = JSON().with_variant(postgresql.JSONB(), "postgresql")


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


def create_schema(metadata: MetaData, engine: Engine) -> None:
    """Create startup tables, tolerating another instance creating them first.

    ``checkfirst`` and ``CREATE`` are separate operations. A competing process
    can create either table between them. Retry the complete existence check in
    a fresh transaction after a duplicate-object error, including on PostgreSQL
    where the failed DDL aborts the preceding transaction.
    """
    for attempt in range(_SCHEMA_CREATE_ATTEMPTS):
        try:
            metadata.create_all(engine, checkfirst=True)
        except DBAPIError as error:
            if (
                attempt == _SCHEMA_CREATE_ATTEMPTS - 1
                or not _is_concurrent_table_creation(error)
            ):
                raise
        else:
            return


def make_metadata() -> MetaData:
    """Return a fresh, empty :class:`~sqlalchemy.MetaData`.

    A new ``MetaData`` per dynamic-table build avoids "table already defined"
    collisions across repeated operations on the same dataset.
    """
    return MetaData()


def build_reference_table(metadata: MetaData, name: str) -> Table:
    """Build the ``trustyai_v2_table_reference`` table (one row per dataset)."""
    return Table(
        name,
        metadata,
        Column("table_idx", _AUTOINCREMENT_PK, primary_key=True, autoincrement=True),
        # UNIQUE guards against a concurrent-create race (two writers inserting
        # the same dataset_name) producing split datasets. Applied by create_all()
        # to new databases only; retrofitting already-deployed tables needs a
        # migration (tracked in issue #335).
        Column("dataset_name", String(255), unique=True),
        Column("metadata", _JSON),
        Column("n_rows", BigInteger),
    )


def build_partial_payload_table(metadata: MetaData, name: str) -> Table:
    """Build the ``trustyai_v2_partial_payloads`` table."""
    return Table(
        name,
        metadata,
        Column("payload_id", String(255)),
        Column("is_input", Boolean),
        Column("payload_data", _BLOB),
    )


def build_dataset_table(
    metadata: MetaData, name: str, cleaned_names: list[str]
) -> Table:
    """Build a dynamic per-dataset table: ``row_idx`` PK + one blob column each.

    :param cleaned_names: SQL-safe column names (``column_0`` .. ``column_n``).
    """
    return Table(
        name,
        metadata,
        Column("row_idx", _AUTOINCREMENT_PK, primary_key=True, autoincrement=True),
        *[Column(col, _BLOB) for col in cleaned_names],
    )
