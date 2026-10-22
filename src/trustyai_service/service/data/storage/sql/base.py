"""Shared SQLAlchemy Core implementation of :class:`StorageInterface`.

``SQLStorage`` owns the full control flow for every SQL backend once: the write
path (reshape, validation, gzip/JSON cell encoding, batch insert, metadata
maintenance, ``n_rows`` upkeep), the read path (ordered ``LIMIT``/``OFFSET``
paging, decode), name-mapping, metadata, payload persistence, dataset lifecycle,
and dynamic table construction. Subclasses supply only an ``Engine`` (and any
dialect override the Core layer cannot infer).

The v2 layout is preserved exactly so existing PostgreSQL/MariaDB data written by
the raw-SQL backends stays readable:

- ``trustyai_v2_table_reference`` -- one row per dataset (``table_idx``,
  ``dataset_name``, JSON ``metadata``, ``n_rows``).
- ``trustyai_v2_partial_payloads`` -- ``payload_id``, ``is_input``, blob
  ``payload_data``.
- ``trustyai_v2_dataset_{idx}`` -- created dynamically per dataset: ``row_idx``
  primary key + one blob column per data column.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import sqlite3
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import Engine, Table, delete, insert, inspect, select, update
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.schema import CreateTable

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sqlalchemy import Connection

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.data.storage.exceptions import DeserializationError
from trustyai_service.service.data.storage.sql import schema
from trustyai_service.service.data.storage.storage_interface import StorageInterface
from trustyai_service.service.serialization import deserialize_model, serialize_model
from trustyai_service.service.serialization.detection import safe_gzip_decompress
from trustyai_service.service.serialization.encoders import (
    json_decoder_hook,
    json_encoder,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

_MIN_MATRIX_NDIM = 2  # Minimum number of dimensions for a 2-D matrix
_MYSQL_DUPLICATE_KEY = 1062
_MYSQL_CONCURRENT_DDL = 1684
_MAX_IMPLICIT_DDL_ATTEMPTS = 3


def require_existing_dataset[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Assert the first non-self argument names an existing dataset."""

    async def validate_dataset_exists(*args: P.args, **kwargs: P.kwargs) -> R:
        storage = args[0]
        # dataset_name is the first non-self argument; support keyword calls too.
        dataset_name = args[1] if len(args) > 1 else kwargs["dataset_name"]
        if not await storage.dataset_exists(dataset_name):
            msg = f"Error when calling {func.__name__}: Dataset '{dataset_name}' does not exist."
            raise ValueError(msg)
        return await func(*args, **kwargs)

    return validate_dataset_exists


def get_clean_column_names(column_names: list[str]) -> list[str]:
    """Generate SQL-safe column names, avoiding injection from real column names."""
    return [f"column_{i}" for i in range(len(column_names))]


class SQLStorage(StorageInterface):
    """SQLAlchemy Core storage base shared by all SQL backends."""

    # Overridden by subclasses; used in DeserializationError messages.
    _backend_name = "SQL"

    def __init__(self, engine: Engine) -> None:
        """Initialize storage against ``engine`` and create the schema tables."""
        self._engine = engine

        self.schema_prefix = "trustyai_v2"
        self.dataset_reference_table = f"{self.schema_prefix}_table_reference"
        self.partial_payload_table = f"{self.schema_prefix}_partial_payloads"

        self._metadata = schema.make_metadata()
        self._ref = schema.build_reference_table(
            self._metadata, self.dataset_reference_table
        )
        self._payloads = schema.build_partial_payload_table(
            self._metadata, self.partial_payload_table
        )
        schema.create_schema(self._metadata, self._engine)

    # === INTERNAL HELPER FUNCTIONS ================================================================
    def _build_table_name(self, index: int) -> str:
        return f"{self.schema_prefix}_dataset_{index}"

    def _dataset_table(self, table_name: str, cleaned_names: list[str]) -> Table:
        """Build a fresh dynamic-table object (own MetaData avoids collisions)."""
        return schema.build_dataset_table(
            schema.make_metadata(), table_name, cleaned_names
        )

    @require_existing_dataset
    async def _get_clean_table_name(self, dataset_name: str) -> str:
        """Get the generated table name for a dataset (SQL-injection-safe)."""
        with self._engine.connect() as conn:
            # ORDER BY table_idx LIMIT 1 makes the lookup deterministic and
            # crash-free even on a legacy DB that predates the dataset_name
            # UNIQUE constraint and already holds duplicate rows.
            idx = conn.execute(
                select(self._ref.c.table_idx)
                .where(self._ref.c.dataset_name == dataset_name)
                .order_by(self._ref.c.table_idx)
                .limit(1)
            ).scalar_one()
        return self._build_table_name(idx)

    @require_existing_dataset
    async def _get_dataset_metadata(self, dataset_name: str) -> dict | None:
        """Return the parsed ``metadata`` document for a dataset."""
        with self._engine.connect() as conn:
            # SQLAlchemy's JSON type returns an already-parsed dict on both
            # PostgreSQL (JSONB) and SQLite/MariaDB (JSON/TEXT).
            return conn.execute(
                select(self._ref.c["metadata"])
                .where(self._ref.c.dataset_name == dataset_name)
                .order_by(self._ref.c.table_idx)
                .limit(1)
            ).scalar_one()

    # === DATASET QUERYING ==========================================================================
    async def dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset exists within the TrustyAI model data."""
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(self._ref.c.dataset_name).where(
                        self._ref.c.dataset_name == dataset_name
                    )
                ).first()
                return row is not None
        except (ProgrammingError, OperationalError):
            # Reference table absent (e.g. after reset_database).
            return False

    def _list_all_datasets_sync(self) -> list[str]:
        with self._engine.connect() as conn:
            return [
                x[0] for x in conn.execute(select(self._ref.c.dataset_name)).fetchall()
            ]

    async def list_all_datasets(self) -> list[str]:
        """List all datasets in the database."""
        return await asyncio.to_thread(self._list_all_datasets_sync)

    @require_existing_dataset
    async def dataset_rows(self, dataset_name: str) -> int:
        """Get the number of rows in a stored dataset (equivalent to data.shape[0])."""
        with self._engine.connect() as conn:
            return conn.execute(
                select(self._ref.c.n_rows)
                .where(self._ref.c.dataset_name == dataset_name)
                .order_by(self._ref.c.table_idx)
                .limit(1)
            ).scalar_one()

    @require_existing_dataset
    async def dataset_cols(self, dataset_name: str) -> int:
        """Get the number of columns in a stored dataset (equivalent to data.shape[1])."""
        table_name = await self._get_clean_table_name(dataset_name)
        columns = inspect(self._engine).get_columns(table_name)
        return len(columns) - 1  # subtract the row_idx primary key

    @require_existing_dataset
    async def dataset_shape(self, dataset_name: str) -> tuple[int]:
        """Get the whole shape of a stored dataset (equivalent to data.shape)."""
        rows = await self.dataset_rows(dataset_name)
        shape = (await self._get_dataset_metadata(dataset_name))["shape"]
        shape[0] = rows
        return tuple(shape)

    # === DATASET READING AND WRITING ===============================================================
    async def write_data(
        self, dataset_name: str, new_rows: np.ndarray, column_names: list[str]
    ) -> None:
        """Write some rows to the database.

        `dataset_name`: the name of the dataset to write to. This is NOT the table name;
                       this should be some string descriptor of the dataset
                       (e.g., model_ABC_input_data).
        `new_rows`: the Numpy array representing the new rows-to-write.
        `column_names`: The corresponding column names within the rows. If appending data,
                       these names must match the existing column names found within
                       `trustyai_v2_table_reference.metadata.column_names`.
        """
        await self.write_data_batch([(dataset_name, new_rows, column_names)])

    @staticmethod
    def _validate_new_rows(new_rows: np.ndarray, column_names: list[str]) -> np.ndarray:
        """Normalize rows and validate their supplied schema without writing."""
        if len(new_rows) == 0:
            msg = f"No data provided! `new_rows`=={new_rows}."
            raise ValueError(msg)

        # if received a single row, reshape into a single-column matrix
        if new_rows.ndim < _MIN_MATRIX_NDIM:
            new_rows = new_rows.reshape(-1, 1)

        # validate that the number of provided column names matches the shape of the provided array
        if new_rows.shape[1] != len(column_names):
            msg = (
                f"Shape mismatch: Number of provided column names ({len(column_names)}) "
                f"does not match number of columns in provided array ({new_rows.shape[1]})."
            )
            raise ValueError(msg)

        return new_rows

    def _validate_dataset_write(
        self,
        conn: Connection,
        dataset_name: str,
        new_rows: np.ndarray,
        column_names: list[str],
        *,
        lock: bool = False,
    ) -> Table | None:
        """Validate an existing dataset using the caller's connection."""
        query = (
            select(self._ref.c.table_idx, self._ref.c["metadata"])
            .where(self._ref.c.dataset_name == dataset_name)
            .order_by(self._ref.c.table_idx)
            .limit(1)
        )
        if lock:
            query = query.with_for_update()
        reference = conn.execute(query).first()
        if reference is None:
            return None

        table_idx, metadata = reference
        stored_shape = metadata["shape"]
        ncols = stored_shape[1]
        if ncols != len(column_names):
            msg = (
                f"Shape mismatch: Number of provided column names ({len(column_names)})"
                f" does not match number of columns in existing database ({ncols})."
            )
            raise ValueError(msg)

        # Cells are stored positionally, so names must match in both value and order.
        stored_names = metadata["column_names"]
        if column_names != stored_names:
            msg = (
                f"Column mismatch: provided column names ({column_names}) do not"
                f" match the column names of the existing database ({stored_names})."
            )
            raise ValueError(msg)
        if list(stored_shape[1:]) != list(new_rows.shape[1:]):
            msg = (
                f"Shape mismatch: new_rows.shape[1:] ({new_rows.shape[1:]}) does not"
                f" match shape of existing database ({stored_shape[1:]})."
            )
            raise ValueError(msg)
        return self._dataset_table(
            self._build_table_name(table_idx), get_clean_column_names(column_names)
        )

    def _create_dataset(
        self,
        conn: Connection,
        dataset_name: str,
        new_rows: np.ndarray,
        column_names: list[str],
    ) -> Table:
        """Create an empty dataset using the caller's connection."""
        metadata = {
            "column_names": column_names,
            "aliased_names": column_names,
            "shape": [-1, *new_rows.shape[1:]],
        }
        result = conn.execute(
            insert(self._ref).values(
                dataset_name=dataset_name, metadata=metadata, n_rows=0
            )
        )
        table_name = self._build_table_name(result.inserted_primary_key[0])
        logger.info(
            "Creating table = %s to store data from %s.", table_name, dataset_name
        )
        ds_table = self._dataset_table(table_name, get_clean_column_names(column_names))
        if self._engine.dialect.name in {"mariadb", "mysql"}:
            # CREATE commits the reference before creating its table. Another
            # writer may observe that reference and finish the same table.
            conn.execute(CreateTable(ds_table, if_not_exists=True))
        else:
            ds_table.create(conn, checkfirst=True)
        return ds_table

    def _is_dataset_creation_conflict(self, error: IntegrityError) -> bool:
        """Recognize only a duplicate dataset name in the reference insert."""
        if (
            not (error.statement or "")
            .lstrip()
            .startswith(f"INSERT INTO {self.dataset_reference_table} ")
        ):
            return False
        original = error.orig
        dialect = self._engine.dialect.name
        if dialect == "postgresql":
            diagnostic = getattr(original, "diag", None)
            return (
                getattr(original, "sqlstate", None) == "23505"
                and getattr(diagnostic, "table_name", None)
                == self.dataset_reference_table
                and getattr(diagnostic, "constraint_name", None)
                == f"{self.dataset_reference_table}_dataset_name_key"
            )
        if dialect in {"mariadb", "mysql"}:
            errno = getattr(original, "errno", None)
            if errno is None and original.args:
                errno = original.args[0]
            message = original.args[1] if len(original.args) > 1 else str(original)
            key = str(message).rpartition(" for key ")[2]
            return errno == _MYSQL_DUPLICATE_KEY and key in {
                "'dataset_name'",
                f"'{self.dataset_reference_table}.dataset_name'",
            }
        return (
            dialect == "sqlite"
            and getattr(original, "sqlite_errorcode", None)
            == sqlite3.SQLITE_CONSTRAINT_UNIQUE
            and str(original)
            == f"UNIQUE constraint failed: {self.dataset_reference_table}.dataset_name"
        )

    def _is_concurrent_ddl_error(self, error: OperationalError) -> bool:
        """Recognize MariaDB/MySQL's concurrent table-definition error."""
        if self._engine.dialect.name not in {"mariadb", "mysql"}:
            return False
        original = error.orig
        errno = getattr(original, "errno", None)
        if errno is None and original.args:
            errno = original.args[0]
        return errno == _MYSQL_CONCURRENT_DDL

    def _prepare_implicit_commit_dataset(
        self,
        dataset_name: str,
        new_rows: np.ndarray,
        column_names: list[str],
    ) -> None:
        """Finish MariaDB/MySQL dataset DDL before any inference row is written."""
        for attempt in range(_MAX_IMPLICIT_DDL_ATTEMPTS):
            try:
                with self._engine.begin() as conn:
                    # Locking a missing reference can take an InnoDB gap lock
                    # and deadlock competing inserts. The unique key arbitrates
                    # creation.
                    table = self._validate_dataset_write(
                        conn, dataset_name, new_rows, column_names
                    )
                    if table is None:
                        self._create_dataset(conn, dataset_name, new_rows, column_names)
                    elif not inspect(conn).has_table(table.name):
                        # A creator's DDL may have committed its reference while
                        # its CREATE is still pending. Both writers can safely
                        # finish it.
                        conn.execute(CreateTable(table, if_not_exists=True))
            except OperationalError as error:
                if (
                    attempt == _MAX_IMPLICIT_DDL_ATTEMPTS - 1
                    or not self._is_concurrent_ddl_error(error)
                ):
                    raise
            else:
                return

    @staticmethod
    def _encode_rows(
        new_rows: np.ndarray, column_names: list[str]
    ) -> list[dict[str, bytes]]:
        """Encode every cell before any part of the batch is written."""
        cleaned_names = get_clean_column_names(column_names)
        params = []
        for new_row in new_rows:
            row_params = {}
            for col_name, col in zip(cleaned_names, new_row, strict=True):
                json_bytes = json.dumps(col, default=json_encoder).encode("utf-8")
                row_params[col_name] = gzip.compress(json_bytes)
            params.append(row_params)

        return params

    async def write_data_batch(
        self, datasets: list[tuple[str, np.ndarray, list[str]]]
    ) -> None:
        """Preflight the whole batch, then commit all rows and counts atomically.

        MariaDB/MySQL implicitly commit DDL. New empty datasets are therefore
        created before the row transaction on those backends, and may remain
        after a failed first write. Existing datasets are never mutated during
        preflight; all inference rows and counts share one transaction.
        """
        datasets = [
            (name, self._validate_new_rows(rows, columns), columns)
            for name, rows, columns in datasets
        ]
        with self._engine.connect() as conn:
            for name, rows, columns in datasets:
                self._validate_dataset_write(conn, name, rows, columns)
        parameters = [self._encode_rows(rows, columns) for _, rows, columns in datasets]

        # A conflict can occur once per newly created dataset. Retry only after
        # the entire failed transaction is rolled back, then revalidate the
        # winning schema and append the complete batch in a fresh transaction.
        for attempt in range(len(datasets) + 1):
            try:
                self._write_data_batch_once(datasets, parameters)
            except IntegrityError as error:
                if attempt == len(datasets) or not self._is_dataset_creation_conflict(
                    error
                ):
                    raise
            else:
                return

    def _write_data_batch_once(
        self,
        datasets: list[tuple[str, np.ndarray, list[str]]],
        parameters: list[list[dict[str, bytes]]],
    ) -> None:
        """Prepare dataset tables, then write every batch row in one transaction."""
        implicit_ddl_commit = self._engine.dialect.name in {"mariadb", "mysql"}
        if implicit_ddl_commit:
            for name, rows, columns in datasets:
                self._prepare_implicit_commit_dataset(name, rows, columns)

        with self._engine.begin() as conn:
            # Revalidate and lock before inserting anything. Concurrent writers
            # must acquire the same reference locks before allocating row IDs,
            # so the order of input/output/metadata rows remains consistent.
            tables = [
                self._validate_dataset_write(conn, name, rows, columns, lock=True)
                for name, rows, columns in datasets
            ]
            if implicit_ddl_commit and any(table is None for table in tables):
                msg = "A dataset was removed while preparing the batch write."
                raise ValueError(msg)
            for (name, rows, columns), table, params in zip(
                datasets, tables, parameters, strict=True
            ):
                ds_table = (
                    self._create_dataset(conn, name, rows, columns)
                    if table is None
                    else table
                )
                conn.execute(insert(ds_table), params)
                conn.execute(
                    update(self._ref)
                    .where(self._ref.c.dataset_name == name)
                    .values(n_rows=self._ref.c.n_rows + len(rows))
                )

    @require_existing_dataset
    async def read_data(
        self, dataset_name: str, start_row: int = 0, n_rows: int | None = None
    ) -> np.ndarray:
        """Read saved data from the database using SQL LIMIT/OFFSET.

        Args:
            dataset_name: The name of the dataset to read (NOT the table name).
                         See trustyai_v2_table_reference.dataset_name or use
                         list_all_datasets() for available dataset names.
            start_row: The row index to start reading from (OFFSET). Defaults to 0.
            n_rows: The number of rows to read (LIMIT). If None, reads all remaining rows.

        Returns:
            NumPy array containing the requested rows.

        """
        table_name = await self._get_clean_table_name(dataset_name)
        ncols = await self.dataset_cols(dataset_name)
        cleaned_names = [f"column_{i}" for i in range(ncols)]
        ds_table = self._dataset_table(table_name, cleaned_names)

        if start_row is None:
            start_row = 0
        if n_rows is None:
            n_rows = await self.dataset_rows(dataset_name)

        with self._engine.connect() as conn:
            rows = conn.execute(
                select(ds_table)
                .order_by(ds_table.c.row_idx.asc())
                .limit(n_rows)
                .offset(start_row)
            ).fetchall()

        # parse saved data back to Numpy array (JSON + gzip)
        arr = []
        dtypes = set()
        for row in rows:
            # first value in row is the index, so we can skip that
            row_values = []
            for cell in row[1:]:
                # coerce blob (bytes or memoryview) to bytes for decompression
                json_str = safe_gzip_decompress(bytes(cell)).decode("utf-8")
                value = np.asarray(json.loads(json_str, object_hook=json_decoder_hook))
                dtypes.add(value.dtype)
                row_values.append(value)
            arr.append(row_values)

        # if all objects have the same dtype, use it, else use object
        return np.array(arr, dtype=dtypes.pop() if len(dtypes) == 1 else object)

    # === COLUMN NAMES =============================================================================
    @require_existing_dataset
    async def get_original_column_names(self, dataset_name: str) -> list[str] | None:
        """Return the original column names for a dataset."""
        return (await self._get_dataset_metadata(dataset_name)).get("column_names")

    @require_existing_dataset
    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        """Return the aliased column names for a dataset."""
        return (await self._get_dataset_metadata(dataset_name)).get("aliased_names")

    @require_existing_dataset
    async def apply_name_mapping(
        self, dataset_name: str, name_mapping: dict[str, str]
    ) -> None:
        """Apply a name mapping to a dataset.

        `dataset_name`: the name of the dataset to read. This is NOT the table name;
            see `trustyai_v2_table_reference.dataset_name` or use list_all_datasets() for the available dataset_names.
        `name_mapping`: a dictionary mapping column names to aliases. Keys should correspond
            to original column names and values should correspond to the desired new names.
        """
        original_names = await self.get_original_column_names(dataset_name)
        aliased_names = await self.get_aliased_column_names(dataset_name)

        # get the new set of optionaly-aliased column names
        for col_idx, original_name in enumerate(original_names):
            # if no match in the mapping, use original name
            aliased_names[col_idx] = name_mapping.get(original_name, original_name)

        await self._set_aliased_names(dataset_name, aliased_names)

    @require_existing_dataset
    async def clear_name_mapping(self, dataset_name: str) -> None:
        """Clear/remove the name mapping for a dataset by resetting aliased_names to original column_names."""
        original_names = await self.get_original_column_names(dataset_name)
        await self._set_aliased_names(dataset_name, original_names)

    async def _set_aliased_names(
        self, dataset_name: str, aliased_names: list[str]
    ) -> None:
        """Read-modify-write the ``aliased_names`` field of the JSON metadata.

        Done in Python rather than via dialect JSON-path functions (``jsonb_set``
        vs ``JSON_SET``) so the code stays fully dialect-agnostic.
        """
        with self._engine.begin() as conn:
            metadata = conn.execute(
                select(self._ref.c["metadata"])
                .where(self._ref.c.dataset_name == dataset_name)
                .order_by(self._ref.c.table_idx)
                .limit(1)
            ).scalar_one()
            metadata["aliased_names"] = aliased_names
            conn.execute(
                update(self._ref)
                .where(self._ref.c.dataset_name == dataset_name)
                .values(metadata=metadata)
            )

    async def get_known_models(self) -> list[str]:
        """Get a list of all model IDs that have inference data stored."""
        all_datasets = await self.list_all_datasets()
        model_ids = set()

        for dataset_name in all_datasets:
            # Skip internal datasets
            if dataset_name.startswith("trustyai_internal_"):
                continue

            # Extract model ID by removing suffixes
            if dataset_name.endswith("_inputs"):
                model_ids.add(dataset_name[: -len("_inputs")])
            elif dataset_name.endswith("_outputs"):
                model_ids.add(dataset_name[: -len("_outputs")])
            elif dataset_name.endswith("_metadata"):
                model_ids.add(dataset_name[: -len("_metadata")])

        return list(model_ids)

    async def get_metadata(self, model_id: str) -> dict:
        """Get metadata for a specific model including shapes, column names, etc."""
        input_dataset = f"{model_id}_inputs"
        output_dataset = f"{model_id}_outputs"
        metadata_dataset = f"{model_id}_metadata"

        metadata = {
            "modelId": model_id,
            "inputData": None,
            "outputData": None,
            "metadataData": None,
        }

        # Get input data metadata
        if await self.dataset_exists(input_dataset):
            try:
                input_shape = await self.dataset_shape(input_dataset)
                input_names = await self.get_original_column_names(input_dataset)
                aliased_input_names = await self.get_aliased_column_names(input_dataset)
                metadata["inputData"] = {
                    "shape": list(input_shape) if input_shape is not None else [],
                    "columnNames": list(input_names) if input_names is not None else [],
                    "aliasedNames": list(aliased_input_names)
                    if aliased_input_names is not None
                    else [],
                }
            except Exception as e:
                logger.warning("Error getting input metadata for %s: %s", model_id, e)

        # Get output data metadata
        if await self.dataset_exists(output_dataset):
            try:
                output_shape = await self.dataset_shape(output_dataset)
                output_names = await self.get_original_column_names(output_dataset)
                aliased_output_names = await self.get_aliased_column_names(
                    output_dataset
                )
                metadata["outputData"] = {
                    "shape": list(output_shape) if output_shape is not None else [],
                    "columnNames": list(output_names)
                    if output_names is not None
                    else [],
                    "aliasedNames": list(aliased_output_names)
                    if aliased_output_names is not None
                    else [],
                }
            except Exception as e:
                logger.warning("Error getting output metadata for %s: %s", model_id, e)

        # Get metadata data info
        if await self.dataset_exists(metadata_dataset):
            try:
                metadata_shape = await self.dataset_shape(metadata_dataset)
                metadata_names = await self.get_original_column_names(metadata_dataset)
                metadata["metadataData"] = {
                    "shape": list(metadata_shape) if metadata_shape is not None else [],
                    "columnNames": list(metadata_names)
                    if metadata_names is not None
                    else [],
                }
            except Exception as e:
                logger.warning("Error getting metadata info for %s: %s", model_id, e)

        return metadata

    # === PARTIAL PAYLOADS =========================================================================
    async def persist_partial_payload(
        self,
        payload: PartialPayload | KServeInferenceRequest | KServeInferenceResponse,
        payload_id: str,
        *,
        is_input: bool,
    ) -> None:
        """Save a partial payload to the database using secure JSON + gzip serialization."""
        with self._engine.begin() as conn:
            conn.execute(
                insert(self._payloads).values(
                    payload_id=payload_id,
                    is_input=is_input,
                    payload_data=serialize_model(payload),
                )
            )

    async def get_partial_payload(
        self, payload_id: str, *, is_input: bool, is_modelmesh: bool
    ) -> PartialPayload | KServeInferenceRequest | KServeInferenceResponse | None:
        """Retrieve a partial payload from the database.

        Uses JSON + gzip deserialization. Returns None if not found.
        """
        with self._engine.connect() as conn:
            result = conn.execute(
                select(self._payloads.c.payload_data).where(
                    (self._payloads.c.payload_id == payload_id)
                    & (self._payloads.c.is_input == is_input)
                )
            ).first()
        if result is None or len(result) == 0:
            # Payload not found in database - this is expected for new payloads
            return None

        # Determine target class based on payload type
        if is_modelmesh:
            target_class = PartialPayload
        elif is_input:  # kserve input
            target_class = KServeInferenceRequest
        else:  # kserve output
            target_class = KServeInferenceResponse

        try:
            # coerce blob (bytes or memoryview) to bytes for deserialization
            return deserialize_model(bytes(result[0]), target_class)
        except Exception as e:
            # Deserialization failure indicates data corruption or format issue
            # This is distinct from "not found" and should be raised to caller
            logger.exception(
                "Deserialization failed for payload '%s' (%s, %s)",
                payload_id,
                "ModelMesh" if is_modelmesh else "KServe",
                "input" if is_input else "output",
            )
            raise DeserializationError(
                payload_id=payload_id,
                reason=f"Failed to deserialize {'ModelMesh' if is_modelmesh else 'KServe'} "
                f"{'input' if is_input else 'output'} payload from {self._backend_name}",
                original_exception=e,
            ) from e

    async def delete_partial_payload(self, payload_id: str, *, is_input: bool) -> None:
        """Delete a partial payload from the database."""
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._payloads).where(
                    (self._payloads.c.payload_id == payload_id)
                    & (self._payloads.c.is_input == is_input)
                )
            )

    async def persist_modelmesh_payload(
        self, payload: PartialPayload, request_id: str, *, is_input: bool
    ) -> None:
        """Persist a ModelMesh partial payload."""
        await self.persist_partial_payload(payload, request_id, is_input=is_input)

    async def get_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> PartialPayload | None:
        """Retrieve a ModelMesh partial payload."""
        return await self.get_partial_payload(
            request_id, is_input=is_input, is_modelmesh=True
        )

    async def delete_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> None:
        """Delete a ModelMesh partial payload."""
        await self.delete_partial_payload(request_id, is_input=is_input)

    # === DATABASE CLEANUP =========================================================================
    @require_existing_dataset
    async def delete_dataset(self, dataset_name: str) -> None:
        """Delete a dataset and its storage table."""
        table_name = await self._get_clean_table_name(dataset_name)
        logger.info("Deleting table=%s to delete dataset=%s.", table_name, dataset_name)
        ds_table = self._dataset_table(table_name, [])
        with self._engine.begin() as conn:
            conn.execute(
                delete(self._ref).where(self._ref.c.dataset_name == dataset_name)
            )
            ds_table.drop(conn, checkfirst=True)

    async def delete_all_datasets(self) -> None:
        """Delete all datasets from the database."""
        for dataset_name in await self.list_all_datasets():
            logger.warning("Deleting dataset %s", dataset_name)
            await self.delete_dataset(dataset_name)

    async def reset_database(self) -> None:
        """Drop all tables and reset the database to a clean state."""
        logger.warning("Fully resetting TrustyAI V2 database.")
        await self.delete_all_datasets()
        self._ref.drop(self._engine, checkfirst=True)
        self._payloads.drop(self._engine, checkfirst=True)
