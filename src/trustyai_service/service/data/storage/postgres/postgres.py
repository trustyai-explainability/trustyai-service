"""PostgreSQL storage backend for TrustyAI inference data."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging

import numpy as np
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.data.storage.exceptions import DeserializationError
from trustyai_service.service.data.storage.postgres.utils import (
    PostgresConnectionManager,
    get_clean_column_names,
    require_existing_dataset,
)
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


class PostgreSQLStorage(StorageInterface):
    """=== v2 DATABASE SCHEMA =========================================================================.

    A faithful port of :class:`MariaDBStorage` to PostgreSQL (via psycopg 3).
    Schema names, metadata shape, method signatures, error messages, and commit
    points match MariaDB exactly; only the SQL dialect and driver differ.

    === Metadata Tables ===
    `trustyai_v2_table_reference`: Reference information about the inference data tables-
                                   e.g., shape, source dataset, etc
        - `table_idx`, BIGINT: Dataset index- this will identify which table is being
                             referenced by this particular row. e.g., if `table_idx=$N`,
                             this row describes the table `trustyai_dataset_$N`
        - `dataset_name`, varchar(255): The name of the dataset stored in
                                       `trustyai_dataset_$TABLE_IDX`
        - `metadata`, JSONB: json of dataset metadata, with schema:
            - `column_names`: The raw column names of the model, straight from the
                             original payloads
            - `aliased_names`: The current state of column aliasing - this will reflect
                              the most recent name-mapping
            - `shape`: The shape of the dataset stored within `trustyai_dataset_$TABLE_IDX`,
                      in the form (-1, x, y, ... z). The row dimension is always -1.
        - `n_rows`, BIGINT: The number of rows within the dataset.
    `trustyai_v2_partial_payloads`: Store partial payloads prior to reconciliation
        - `payload_id`, varchar(255): The id of the partial payload
        - `is_input`, BOOLEAN: Whether the partial payload is an input or output payload
        - `payload_data`, BYTEA: The serialized partial payload (JSON + gzip)

    === Inference Data Tables ===
    Each dataset is stored in its own table named `trustyai_v2_dataset_X`, where `X` is
    an incrementing integer assigned by the DB

    `trustyai_v2_dataset_X`: stores the data for dataset_X. Information about dataset_X
                            can be found in `trustyai_v2_table_reference` in the row
                            where `table_idx`==`X`
     - `column_0`, BYTEA: the serialized data for the 0th column of this row, e.g., arr[$row][0]
     - `column_1`, BYTEA: the serialized data for the 1st column of this row, e.g., arr[$row][1]
     - ...
     - `column_n`, BYTEA: the serialized data for the final column of this row, e.g., arr[$row][n]

    === SQL INJECTION SAFETY ======================================================================
    All dynamic identifiers (table names, `column_N` names) are composed using
    ``psycopg.sql.Identifier`` (double-quote quoting); all user-provided values
    (dataset_name, payload_id, etc.) are always passed as parameterized ``%s``
    query arguments. Dataset table names are constructed from auto-incrementing
    integer indices via ``_build_table_name()``, never from user-provided dataset
    names directly.
    """

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        database: str,
        *,
        ssl_ca: str | None = None,
    ) -> None:
        """Initialize PostgreSQL storage and create schema tables."""
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.connection_manager = PostgresConnectionManager(
            user, password, host, port, database, ssl_ca=ssl_ca
        )

        self.schema_prefix = "trustyai_v2"
        self.dataset_reference_table = f"{self.schema_prefix}_table_reference"
        # stores partial payloads
        self.partial_payload_table = f"{self.schema_prefix}_partial_payloads"

        with self.connection_manager as (conn, cursor):
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {table} "
                    "(table_idx BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                    "dataset_name varchar(255), metadata JSONB, n_rows BIGINT)"
                ).format(table=sql.Identifier(self.dataset_reference_table))
            )
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {table} "
                    "(payload_id varchar(255), is_input BOOLEAN, payload_data BYTEA)"
                ).format(table=sql.Identifier(self.partial_payload_table))
            )
            # PostgreSQL DDL is transactional and does not implicitly commit.
            conn.commit()

    # === INTERNAL HELPER FUNCTIONS ================================================================
    def _build_table_name(self, index: int) -> str:
        return f"{self.schema_prefix}_dataset_{index}"

    @require_existing_dataset
    async def _get_clean_table_name(self, dataset_name: str) -> str:
        """Get a generated table name corresponding to a particular dataset.

        This avoids possible SQL injection from within the model names.
        """
        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                sql.SQL("SELECT table_idx FROM {table} WHERE dataset_name=%s").format(
                    table=sql.Identifier(self.dataset_reference_table)
                ),
                (dataset_name,),
            )
            return self._build_table_name(cursor.fetchone()[0])

    @require_existing_dataset
    async def _get_dataset_metadata(self, dataset_name: str) -> dict | None:
        """Return the metadata field from a particular dataset within the dataset_reference_table."""
        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                sql.SQL("SELECT metadata FROM {table} WHERE dataset_name=%s").format(
                    table=sql.Identifier(self.dataset_reference_table)
                ),
                (dataset_name,),
            )
            # psycopg's JSONB loader returns an already-parsed dict.
            return cursor.fetchone()[0]

    # === DATASET QUERYING ==========================================================================

    async def dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset exists within the TrustyAI model data."""
        try:
            with self.connection_manager as (_conn, cursor):
                cursor.execute(
                    sql.SQL(
                        "SELECT dataset_name FROM {table} WHERE dataset_name=%s"
                    ).format(table=sql.Identifier(self.dataset_reference_table)),
                    (dataset_name,),
                )
                return cursor.fetchone() is not None
        except psycopg.errors.ProgrammingError:
            return False

    def _list_all_datasets_sync(self) -> list[str]:
        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                sql.SQL("SELECT dataset_name FROM {table}").format(
                    table=sql.Identifier(self.dataset_reference_table)
                )
            )
            return [x[0] for x in cursor.fetchall()]

    async def list_all_datasets(self) -> list[str]:
        """List all datasets in the database."""
        return await asyncio.to_thread(self._list_all_datasets_sync)

    @require_existing_dataset
    async def dataset_rows(self, dataset_name: str) -> int:
        """Get the number of rows in a stored dataset (equivalent to data.shape[0])."""
        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                sql.SQL("SELECT n_rows FROM {table} WHERE dataset_name=%s").format(
                    table=sql.Identifier(self.dataset_reference_table)
                ),
                (dataset_name,),
            )
            return cursor.fetchone()[0]

    @require_existing_dataset
    async def dataset_cols(self, dataset_name: str) -> int:
        """Get the number of columns in a stored dataset (equivalent to data.shape[1])."""
        table_name = await self._get_clean_table_name(dataset_name)
        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name=%s AND table_schema=current_schema()",
                (table_name,),
            )
            return cursor.fetchone()[0] - 1

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

        # if this is the first time we've seen this dataset, set up its tables inside the DB
        if not await self.dataset_exists(dataset_name):
            with self.connection_manager as (conn, cursor):
                # create an entry in `trustyai_v2_table_reference`
                metadata = {
                    "column_names": column_names,
                    "aliased_names": column_names,
                    "shape": (-1, *new_rows.shape[1:]),
                }
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {table} (dataset_name, metadata, n_rows) "
                        "VALUES (%s, %s, 0)"
                    ).format(table=sql.Identifier(self.dataset_reference_table)),
                    (dataset_name, Jsonb(metadata)),
                )

                # retrieve the DB-provided table index, to get an SQL-safe name for the dataset storage table
                cursor.execute(
                    sql.SQL(
                        "SELECT table_idx FROM {table} WHERE dataset_name=%s"
                    ).format(table=sql.Identifier(self.dataset_reference_table)),
                    (dataset_name,),
                )
                table_name = self._build_table_name(cursor.fetchone()[0])

                # create SQL-safe column names for the dataset storage table
                cleaned_names = get_clean_column_names(column_names)
                column_defs = sql.SQL(", ").join(
                    sql.SQL("{} BYTEA").format(sql.Identifier(name))
                    for name in cleaned_names
                )

                # create the dataset storage table for this dataset
                logger.info(
                    "Creating table = %s to store data from %s.",
                    table_name,
                    dataset_name,
                )
                cursor.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {table} "
                        "(row_idx BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                        "{cols})"
                    ).format(table=sql.Identifier(table_name), cols=column_defs)
                )

                # Commit everything together. This is to make sure we don't create an orphan
                # DB entry if one of the steps fails
                conn.commit()
            ncols = len(column_names)
            nrows = 0
        else:
            # if dataset already exists, grab its current shape and information
            stored_shape = await self.dataset_shape(dataset_name)
            ncols = stored_shape[1]
            nrows = await self.dataset_rows(dataset_name)
            table_name = await self._get_clean_table_name(dataset_name)
            cleaned_names = get_clean_column_names(column_names)

            # validate that the number of columns in the saved DB matched the provided column names
            if ncols != len(column_names):
                msg = (
                    f"Shape mismatch: Number of provided column names ({len(column_names)})"
                    f" does not match number of columns in existing database ({ncols})."
                )
                raise ValueError(msg)

            # validate that the shape of the inbound data is compatible with the stored data shape
            if list(stored_shape[1:]) != list(new_rows.shape[1:]):
                msg = (
                    f"Shape mismatch: new_rows.shape[1:] ({new_rows.shape[1:]}) does not"
                    f" match shape of existing database ({stored_shape[1:]})."
                )
                raise ValueError(msg)

        with self.connection_manager as (conn, cursor):
            # write each new_rows[i, j] to bytes (JSON + gzip)
            byte_matrix = []
            for new_row in new_rows:
                col_values = []
                for col in new_row:
                    json_bytes = json.dumps(col, default=json_encoder).encode("utf-8")
                    col_values.append(gzip.compress(json_bytes))
                byte_matrix.append(tuple(col_values))

            # place the byte_matrix into the DB (psycopg maps Python bytes -> bytea)
            insert_stmt = sql.SQL(
                "INSERT INTO {table} ({cols}) VALUES ({placeholders})"
            ).format(
                table=sql.Identifier(table_name),
                cols=sql.SQL(", ").join(sql.Identifier(name) for name in cleaned_names),
                placeholders=sql.SQL(", ").join(
                    sql.Placeholder() for _ in cleaned_names
                ),
            )
            cursor.executemany(insert_stmt, byte_matrix)
            cursor.execute(
                sql.SQL("UPDATE {table} SET n_rows=%s WHERE dataset_name=%s").format(
                    table=sql.Identifier(self.dataset_reference_table)
                ),
                (
                    nrows + len(new_rows),
                    dataset_name,
                ),
            )

            # commit as one single transaction
            conn.commit()

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

        if n_rows is None:
            n_rows = await self.dataset_rows(dataset_name)

        with self.connection_manager as (_conn, cursor):
            # grab matching data - using LIMIT/OFFSET for better SQL compatibility
            cursor.execute(
                sql.SQL(
                    "SELECT * FROM {table} ORDER BY row_idx ASC LIMIT %s OFFSET %s"
                ).format(table=sql.Identifier(table_name)),
                (n_rows, start_row),
            )

            # parse saved data back to Numpy array (JSON + gzip)
            arr = []
            dtypes = set()
            for row in cursor.fetchall():
                # first value in row is the index, so we can skip that
                row_values = []
                for cell in row[1:]:
                    # coerce bytea (bytes or memoryview) to bytes for decompression
                    json_str = safe_gzip_decompress(bytes(cell)).decode("utf-8")
                    value = np.asarray(
                        json.loads(json_str, object_hook=json_decoder_hook)
                    )
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

        # overwrite the aliased_names
        with self.connection_manager as (conn, cursor):
            cursor.execute(
                sql.SQL(
                    "UPDATE {table} "
                    "SET metadata = jsonb_set(metadata, '{{aliased_names}}', %s::jsonb) "
                    "WHERE dataset_name=%s"
                ).format(table=sql.Identifier(self.dataset_reference_table)),
                (Jsonb(aliased_names), dataset_name),
            )
            conn.commit()

    @require_existing_dataset
    async def clear_name_mapping(self, dataset_name: str) -> None:
        """Clear/remove the name mapping for a dataset by resetting aliased_names to original column_names."""
        original_names = await self.get_original_column_names(dataset_name)

        # Reset aliased_names to the original column_names
        with self.connection_manager as (conn, cursor):
            cursor.execute(
                sql.SQL(
                    "UPDATE {table} "
                    "SET metadata = jsonb_set(metadata, '{{aliased_names}}', %s::jsonb) "
                    "WHERE dataset_name=%s"
                ).format(table=sql.Identifier(self.dataset_reference_table)),
                (Jsonb(original_names), dataset_name),
            )
            conn.commit()

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
                model_id = dataset_name[: -len("_inputs")]
                model_ids.add(model_id)
            elif dataset_name.endswith("_outputs"):
                model_id = dataset_name[: -len("_outputs")]
                model_ids.add(model_id)
            elif dataset_name.endswith("_metadata"):
                model_id = dataset_name[: -len("_metadata")]
                model_ids.add(model_id)

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
        with self.connection_manager as (conn, cursor):
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {table} (payload_id, is_input, payload_data) "
                    "VALUES (%s, %s, %s)"
                ).format(table=sql.Identifier(self.partial_payload_table)),
                (payload_id, is_input, serialize_model(payload)),
            )
            conn.commit()

    async def get_partial_payload(
        self, payload_id: str, *, is_input: bool, is_modelmesh: bool
    ) -> PartialPayload | KServeInferenceRequest | KServeInferenceResponse | None:
        """Retrieve a partial payload from the database.

        Uses JSON + gzip deserialization. Returns None if not found.
        """
        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                sql.SQL(
                    "SELECT payload_data FROM {table} "
                    "WHERE payload_id=%s AND is_input=%s"
                ).format(table=sql.Identifier(self.partial_payload_table)),
                (payload_id, is_input),
            )
            result = cursor.fetchone()
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
            # coerce bytea (bytes or memoryview) to bytes for deserialization
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
                f"{'input' if is_input else 'output'} payload from PostgreSQL",
                original_exception=e,
            ) from e

    async def delete_partial_payload(self, payload_id: str, *, is_input: bool) -> None:
        """Delete a partial payload from the database."""
        with self.connection_manager as (conn, cursor):
            cursor.execute(
                sql.SQL(
                    "DELETE FROM {table} WHERE payload_id=%s AND is_input=%s"
                ).format(table=sql.Identifier(self.partial_payload_table)),
                (payload_id, is_input),
            )
            conn.commit()

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
        with self.connection_manager as (conn, cursor):
            cursor.execute(
                sql.SQL("DELETE FROM {table} WHERE dataset_name=%s").format(
                    table=sql.Identifier(self.dataset_reference_table)
                ),
                (dataset_name,),
            )
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {table}").format(
                    table=sql.Identifier(table_name)
                )
            )
            conn.commit()

    async def delete_all_datasets(self) -> None:
        """Delete all datasets from the database."""
        for dataset_name in await self.list_all_datasets():
            logger.warning("Deleting dataset %s", dataset_name)
            await self.delete_dataset(dataset_name)

    async def reset_database(self) -> None:
        """Drop all tables and reset the database to a clean state."""
        logger.warning("Fully resetting TrustyAI V2 database.")
        await self.delete_all_datasets()
        with self.connection_manager as (conn, cursor):
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {table}").format(
                    table=sql.Identifier(self.dataset_reference_table)
                )
            )
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {table}").format(
                    table=sql.Identifier(self.partial_payload_table)
                )
            )
            conn.commit()
