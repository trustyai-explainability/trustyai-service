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
        Column("dataset_name", String(255)),
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
