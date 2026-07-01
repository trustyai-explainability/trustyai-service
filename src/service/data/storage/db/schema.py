"""SQLAlchemy Core table definitions for the unified database storage backend."""

from sqlalchemy import (
    Boolean,
    Column,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
)
from sqlalchemy.types import JSON

metadata_obj = MetaData()

datasets_table = Table(
    "trustyai_datasets",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset_name", String(255), unique=True, nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("n_rows", Integer, nullable=False, server_default="0"),
)

data_table = Table(
    "trustyai_data",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset_name", String(255), nullable=False),
    Column("row_idx", Integer, nullable=False),
    Column("row_data", LargeBinary, nullable=False),
    Index("ix_trustyai_data_dataset_row", "dataset_name", "row_idx"),
)

partial_payloads_table = Table(
    "trustyai_partial_payloads",
    metadata_obj,
    Column("payload_id", String(255), nullable=False),
    Column("is_input", Boolean, nullable=False),
    Column("payload_data", LargeBinary, nullable=False),
    Index("ix_trustyai_payloads_id_input", "payload_id", "is_input", unique=True),
)
