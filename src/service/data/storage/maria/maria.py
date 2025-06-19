import asyncio
import io
import json
import logging
import mariadb
import numpy as np
import pickle as pkl
from typing import Optional, Dict, List

from src.service.data.modelmesh_parser import PartialPayload
from src.service.data.storage import LegacyMariaDBStorageReader
from src.service.data.storage.maria.utils import MariaConnectionManager, require_existing_dataset, \
    get_clean_column_names
from src.service.data.storage.storage_interface import StorageInterface

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class MariaDBStorage(StorageInterface):
    """
    === v2 DATABASE SCHEMA =========================================================================

    === Metadata Tables ===
    `trustyai_v2_table_reference`: Reference information about the inference data tables- e.g., shape, source dataset, etc
        - `table_idx`, BIGINT: Dataset index- this will identify which table is being referenced by this particular row.
                             e.g., if `table_idx=$N`, this row describes the table `trustyai_dataset_$N`
        - `dataset_name`, varchar(255): The name of the dataset stored in `trustyai_dataset_$TABLE_IDX`
        - `metadata`, JSON: json of dataset metadata, with schema:
            - `column_names`: The raw column names of the model, straight from the original payloads
            - `aliased_names`: The current state of column aliasing - this will reflect the most recent name-mapping
            - `shape`: The shape of the dataset stored within `trustyai_dataset_$TABLE_IDX`, in the form (-1, x, y, ... z).
                       The row dimension is always represented as -1.
        - `n_rows`, BIGINT: The number of rows within the dataset.
    `trustyai_v2_partial_payloads`: Store partial payloads prior to reconciliation
        - `payload_id`, varchar(255): The id of the partial payload
        - `is_input`, BOOLEAN: Whether the partial payload is an input or output payload
        - `payload_data`, LONGBLOB: The pickled partial payload

    === Inference Data Tables ===
    Each dataset is stored in its own table named `trustyai_v2_dataset_X`, where `X` is an incrementing integer assigned by the DB

    `trustyai_v2_dataset_X`: stores the data for dataset_X. Information about dataset_X can be found in `trustyai_v2_table_reference` in the row where `table_idx`==`X`
     - `column_0`, LONGBLOB: the pickled data for the 0th column of this row, e.g., arr[$row][0]
     - `column_1`, LONGBLOB: the pickled data for the 1st column of this row, e.g., arr[$row][1]
     - ...
     - `column_n`, LONGBLOB: the pickled data for the final column of this row, e.g., arr[$row][n]
    """


    def __init__(self, user: str, password: str, host: str, port: int, database: str, attempt_migration=True):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.connection_manager = MariaConnectionManager(user, password, host, port, database)

        self.schema_prefix = "trustyai_v2"
        self.dataset_reference_table = f"{self.schema_prefix}_table_reference"
        self.partial_payload_table = f"{self.schema_prefix}_partial_payloads" # stores partial payloads

        with self.connection_manager as (conn, cursor):
            cursor.execute(f"CREATE TABLE IF NOT EXISTS `{self.dataset_reference_table}` (table_idx BIGINT AUTO_INCREMENT, dataset_name varchar(255), metadata JSON, n_rows BIGINT, PRIMARY KEY (table_idx))")
            cursor.execute(f"CREATE TABLE IF NOT EXISTS `{self.partial_payload_table}` (payload_id varchar(255), is_input BOOLEAN, payload_data LONGBLOB)")

        if attempt_migration:
            self._migrate_from_legacy_db()

    # === MIGRATORS ================================================================================
    def _migrate_from_legacy_db(self):
        legacy_reader = LegacyMariaDBStorageReader(user=self.user, password=self.password, host=self.host, port=self.port, database=self.database)
        if legacy_reader.legacy_data_exists():
            logger.info("Legacy TrustyAI v1 data exists in database, checking if a migration is necessary.")
            asyncio.run(legacy_reader.migrate_data(self))


    # === INTERNAL HELPER FUNCTIONS ================================================================
    def _build_table_name(self, index):
        return f"{self.schema_prefix}_dataset_{index}"

    @require_existing_dataset
    def _get_clean_table_name(self, dataset_name: str) -> str:
        """
        Get a generated table name corresponding to a particular dataset.
        This avoids possible SQL injection from within the model names.
        """
        with self.connection_manager as (conn, cursor):
            cursor.execute(f"SELECT table_idx FROM `{self.dataset_reference_table}` WHERE dataset_name=?", (dataset_name,))
            return self._build_table_name(cursor.fetchone()[0])


    @require_existing_dataset
    def _get_dataset_metadata(self, dataset_name: str) -> Optional[Dict]:
        """
        Return the metadata field from a particular dataset within the dataset_reference_table.
        """
        with self.connection_manager as (conn, cursor):
            cursor.execute(f"SELECT metadata FROM `{self.dataset_reference_table}` WHERE dataset_name=?", (dataset_name,))
            metadata = cursor.fetchone()[0]
        return json.loads(metadata)


    #=== DATASET QUERYING ==========================================================================
    def dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset exists within the TrustyAI model data.
        """
        try:
            with self.connection_manager as (conn, cursor):
                cursor.execute(f"SELECT dataset_name FROM `{self.dataset_reference_table}` WHERE dataset_name=?", (dataset_name,))
                return cursor.fetchone() is not None
        except mariadb.ProgrammingError:
            return False

    def list_all_datasets(self):
        """
        List all available datasets in the database.
        """
        with self.connection_manager as (conn, cursor):
            cursor.execute(f"SELECT dataset_name FROM `{self.dataset_reference_table}`")
            results = [x[0] for x in cursor.fetchall()]
        return results


    @require_existing_dataset
    def dataset_rows(self, dataset_name: str) -> int:
        """
        Get the number of rows in a stored dataset (equivalent to data.shape[0])
        """

        with self.connection_manager as (conn, cursor):
            cursor.execute(f"SELECT n_rows FROM `{self.dataset_reference_table}` WHERE dataset_name=?", (dataset_name,))
            return cursor.fetchone()[0]


    @require_existing_dataset
    def dataset_cols(self, dataset_name: str) -> int:
        """
        Get the number of columns in a stored dataset (equivalent to data.shape[1])
        """
        table_name = self._get_clean_table_name(dataset_name)
        with self.connection_manager as (conn, cursor):
            cursor.execute(f"SHOW COLUMNS FROM {table_name}")
            return len(cursor.fetchall()) - 1


    @require_existing_dataset
    def dataset_shape(self, dataset_name: str) -> tuple[int]:
        """
        Get the whole shape of a stored dataset (equivalent to data.shape)
        """
        rows = self.dataset_rows(dataset_name)
        shape = self._get_dataset_metadata(dataset_name)["shape"]
        shape[0] = rows
        return tuple(shape)



    # === DATASET READING AND WRITING ===============================================================
    async def write_data(self, dataset_name: str, new_rows: np.ndarray, column_names: List[str]):
        """
        Write some rows to the database

        `dataset_name`: the name of the dataset to write to. This is NOT the table name; this should
            be some string descriptor of the dataset (e.g., model_ABC_input_data).`
        `new_rows`: the Numpy array representing the new rows-to-write.
        `column_names`: The corresponding column names within the rows. If appending data to the table,
            these names must match the existing column names found within `trustyai_v2_table_reference.metadata.column_names`.
        """

        if len(new_rows) == 0:
            raise ValueError(f"No data provided! `new_rows`=={new_rows}.")

        # if received a single row, reshape into a single-row matrix
        if new_rows.ndim < 2:
            new_rows = new_rows.reshape(1, -1)

        # validate that the number of provided column names matches the shape of the provided array
        if new_rows.shape[1] != len(column_names):
            raise ValueError(
                f"Shape mismatch: Number of provided column names ({len(column_names)}) does not match number of columns in provided array ({new_rows.shape[1]}).")

        # if this is the first time we've seen this dataset, set up its tables inside the DB
        if not self.dataset_exists(dataset_name):
            with self.connection_manager as (conn, cursor):

                # create an entry in `trustyai_v2_table_reference`
                metadata = {"column_names": column_names, "aliased_names": column_names, "shape": (-1, *new_rows.shape[1:])}
                cursor.execute(f"INSERT INTO `{self.dataset_reference_table}` (dataset_name, metadata, n_rows) VALUES (?, ?, 0)",
                               (dataset_name, json.dumps(metadata)))

                # retrieve the DB-provided table index, to get an SQL-safe name for the dataset storage table
                cursor.execute(f"SELECT table_idx FROM `{self.dataset_reference_table}` WHERE dataset_name=?", (dataset_name,))
                table_name = self._build_table_name(cursor.fetchone()[0])

                # create SQL-safe column names for the dataset storage table
                cleaned_names = get_clean_column_names(column_names)
                column_name_creator= ", ".join([f"{name} LONGBLOB" for name in cleaned_names])

                # create the dataset storage table for this dataset
                logger.info(f"Creating table = {table_name} to store data from {dataset_name}.")
                cursor.execute(
                    f"CREATE TABLE IF NOT EXISTS `{table_name}` (row_idx BIGINT AUTO_INCREMENT, "
                    f"{column_name_creator}, PRIMARY KEY (row_idx))"
                )

                # Commit everything together. This is to make sure we don't create an orphan
                # DB entry if one of the steps fails
                conn.commit()
            ncols = len(column_names)
            nrows = 0
        else:
            # if dataset already exists, grab its current shape and information
            stored_shape = self.dataset_shape(dataset_name)
            ncols = stored_shape[1]
            nrows = self.dataset_rows(dataset_name)
            table_name = self._get_clean_table_name(dataset_name)
            cleaned_names = get_clean_column_names(column_names)

            # validate that the number of columns in the saved DB matched the provided column names
            if ncols != len(column_names):
                raise ValueError(f"Shape mismatch: Number of provided column names ({len(column_names)})"
                                 f" does not match number of columns in existing database ({ncols}).")

            # validate that the shape of the inbound data is compatible with the stored data shape
            if list(stored_shape[1:]) != list(new_rows.shape[1:]):
                raise ValueError(f"Shape mismatch: new_rows.shape[1:] ({new_rows.shape[1:]}) does not"
                                 f" match shape of existing database ({stored_shape[1:]}).")

        value_formatter = ",".join(["?" for _ in range(ncols)])
        with self.connection_manager as (conn, cursor):
            # write each new_rows[i, j] to bytes
            byte_matrix = []
            for new_row in new_rows:
                col_values = []
                for col in new_row:
                    with io.BytesIO() as bio:
                        np.save(bio, col, allow_pickle=True)
                        col_values.append(bio.getvalue())
                byte_matrix.append(tuple(col_values))

            # place the byte_matrix into the DB
            cursor.executemany(
                f"INSERT INTO `{table_name}` ({','.join(cleaned_names)}) VALUES ({value_formatter})",
                    byte_matrix
            )
            cursor.execute(
                f"UPDATE `{self.dataset_reference_table}` SET n_rows=? WHERE dataset_name=?",
                (nrows+len(new_rows), dataset_name, )
            )

            # commit as one single transaction
            conn.commit()


    @require_existing_dataset
    def read_data(self, dataset_name: str, start_row: int = 0, n_rows: int = None):
        """
        Read saved data from the database, from `start_row` to `start_row + n_rows` (inclusive)

        `dataset_name`: the name of the dataset to read. This is NOT the table name;
            see `trustyai_v2_table_reference.dataset_name` or use list_all_datasets() for the available dataset_names.
        `start_row`: The row to start reading from. If not specified, read from row 0.
        `n_rows`: The total number of rows to read. If not specified, read all rows.

        """
        table_name = self._get_clean_table_name(dataset_name)

        if n_rows is None:
            n_rows = self.dataset_rows(dataset_name)

        with self.connection_manager as (conn, cursor):
            # grab matching data
            cursor.execute(
                f"SELECT * FROM `{table_name}` WHERE row_idx>? AND row_idx<=?",
                (start_row, start_row+n_rows)
            )

            # parse saved data back to Numpy array
            arr = []
            dtypes = set()
            for row in cursor.fetchall():
                # first value in row is the index, so we can skip that
                row_values = []
                for cell in row[1:]:
                    value = np.load(io.BytesIO(cell), allow_pickle=True)

                    dtypes.add(value.dtype)
                    row_values.append(value)
                arr.append(row_values)

            # if all objects have the same dtype, use it, else use object
            arr = np.array(arr, dtype=dtypes.pop() if len(dtypes)==1 else object)
            return arr


    # === COLUMN NAMES =============================================================================
    @require_existing_dataset
    def get_original_column_names(self, dataset_name: str) -> Optional[List[str]]:
        return self._get_dataset_metadata(dataset_name).get("column_names")

    @require_existing_dataset
    def get_aliased_column_names(self, dataset_name: str) -> List[str]:
        return self._get_dataset_metadata(dataset_name).get("aliased_names")

    @require_existing_dataset
    def apply_name_mapping(self, dataset_name: str, name_mapping: Dict[str, str]):
        """Apply a name mapping to a dataset.

        `dataset_name`: the name of the dataset to read. This is NOT the table name;
            see `trustyai_v2_table_reference.dataset_name` or use list_all_datasets() for the available dataset_names.
        `name_mapping`: a dictionary mapping column names to aliases. Keys should correspond
            to original column names and values should correspond to the desired new names.
        """

        original_names = self.get_original_column_names(dataset_name)
        aliased_names = self.get_aliased_column_names(dataset_name)

        # get the new set of optionaly-aliased column names
        for col_idx, original_name in enumerate(original_names):
            # if no match in the mapping, use original name
            aliased_names[col_idx] = name_mapping.get(original_name, original_name)

        # overwrite the aliased_names
        with self.connection_manager as (conn, cursor):
            # parse aliased_name list into parameterized JSON_ARRAY argument
            array_parameters = ", ".join(["?" for _ in aliased_names])
            cursor.execute(
                f"UPDATE `{self.dataset_reference_table}` SET metadata=JSON_SET(metadata, '$.aliased_names', JSON_ARRAY({array_parameters})) WHERE dataset_name=?",
                (*aliased_names, dataset_name,))
            conn.commit()



    # === PARTIAL PAYLOADS =========================================================================
    async def _persist_payload(self, payload, is_input: bool, request_id: Optional[str] = None):
        """Save a partial payload to the database."""
        with self.connection_manager as (conn, cursor):
            if request_id is None:
                request_id = payload.id

            cursor.execute(
                f"INSERT INTO `{self.partial_payload_table}` (payload_id, is_input, payload_data) VALUES (?, ?, ?)",
                (request_id, is_input, pkl.dumps(payload)))
            conn.commit()

    async def _get_partial_payload(self, payload_id: str, is_input: bool):
        """Retrieve a partial payload from the database."""
        with self.connection_manager as (conn, cursor):
            cursor.execute(f"SELECT payload_data FROM `{self.partial_payload_table}` WHERE payload_id=? AND is_input=?", (payload_id, is_input))
            result = cursor.fetchone()
        if result is None or len(result) == 0:
            return None
        payload_data = result[0]
        return pkl.loads(payload_data)


    async def persist_partial_payload(self, payload, is_input: bool):
        await self._persist_payload(payload, is_input)


    async def persist_modelmesh_payload(self, payload: PartialPayload, request_id: str, is_input: bool):
        await self._persist_payload(payload, is_input, request_id=request_id)


    async def get_partial_payload(self, payload_id: str, is_input: bool):
        return await self._get_partial_payload(payload_id, is_input)


    async def get_modelmesh_payload(self, request_id: str, is_input: bool) -> Optional[PartialPayload]:
        return await self._get_partial_payload(request_id, is_input)


    async def delete_modelmesh_payload(self, request_id: str, is_input: bool):
        with self.connection_manager as (conn, cursor):
            cursor.execute(f"DELETE FROM {self.partial_payload_table} WHERE payload_id=? AND is_input=?", (request_id, is_input))
            conn.commit()


    # === DATABASE CLEANUP =========================================================================
    @require_existing_dataset
    def delete_dataset(self, dataset_name: str):
        table_name = self._get_clean_table_name(dataset_name)
        with self.connection_manager as (conn, cursor):
            cursor.execute(f"DELETE FROM `{self.dataset_reference_table}` WHERE dataset_name=?", (dataset_name,))
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            conn.commit()

    def delete_all_datasets(self):
        for dataset_name in self.list_all_datasets():
            logger.warning(f"Deleting dataset {dataset_name}")
            self.delete_dataset(dataset_name)

    def reset_database(self):
        logger.warning(f"Fully resetting TrustyAI V2 database.")
        self.delete_all_datasets()
        with self.connection_manager as (conn, cursor):
            cursor.execute(f"DROP TABLE IF EXISTS `{self.dataset_reference_table}`")
            cursor.execute(f"DROP TABLE IF EXISTS `{self.partial_payload_table}`")
            conn.commit()

