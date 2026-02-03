import javaobj
import logging
import pandas as pd
from typing import List, Tuple
import uuid

from src.service.data.storage.maria.utils import MariaConnectionManager
from src.service.constants import *
from src.service.data.storage.storage_interface import StorageInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class LegacyMariaDBStorageReader:
    def __init__(self, user, password, host, port, database):
        self.connection_manager = MariaConnectionManager(user, password, host, port, database)

    def legacy_data_exists(self):
        with self.connection_manager as (conn, cursor):
            cursor.execute("SHOW TABLES LIKE 'DataframeMetadata'")
            return len(cursor.fetchall()) != 0

    def list_datasets(self):
        """List all legacy datasets in the DB"""
        with self.connection_manager as (conn, cursor):
            cursor.execute("SELECT DISTINCT id" + " FROM DataframeMetadata")
            return [x[0] for x in cursor]

    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if a legacy dataset exists"""
        with self.connection_manager as (conn, cursor):
            cursor.execute("SELECT modelId" + " FROM DataframeRow" + " WHERE modelId=?" + " LIMIT 1;", (dataset_name,))
            return cursor.fetchone() is not None

    def dataset_rows(self, dataset_name: str) -> int:
        """Count the rows in a legacy dataset"""
        with self.connection_manager as (conn, cursor):
            cursor.execute("SELECT COUNT(rowId)" + " FROM DataframeRow" + " WHERE modelId=?", (dataset_name,))
            return cursor.fetchone()[0]

    def _get_column_names(self, dataset_name: str) -> List[str]:
        """Get the column names of a legacy dataset"""
        with self.connection_manager as (conn, cursor):
            cursor.execute(
                "SELECT names" + " FROM DataframeMetadata_names" + " WHERE DataframeMetadata_id=?", (dataset_name,)
            )
            return [x[0] for x in cursor]

    def _get_input_and_output_columns(self, dataset_name: str, column_names: List[str]) -> Tuple[List[str], List[str]]:
        """Get the input and output column names of a legacy dataset"""
        with self.connection_manager as (conn, cursor):
            cursor.execute(
                "SELECT inputs" + " FROM DataframeMetadata_inputs" + " WHERE DataframeMetadata_id=?", (dataset_name,)
            )
            input_columns = []
            output_columns = []
            for i, x in enumerate(cursor):
                if x[0] == b"\x01":
                    input_columns.append(column_names[i])
                else:
                    output_columns.append(column_names[i])

            return input_columns, output_columns

    def _get_name_mapping(self, dataset_name, input_names, output_names):
        """Read name mappings from legacy dataset"""

        with self.connection_manager as (conn, cursor):
            cursor.execute(
                "SELECT items_KEY, mappedItems_KEY"
                + " FROM StorageMetadata_StorageSchema"
                + " JOIN (storageSchema_mappedItems) ON (storageSchema_mappedItems.Schema_id = StorageMetadata_StorageSchema.schemas_id)"
                + " JOIN (storageSchema_originalItems) ON (storageSchema_originalItems.items_id = storageSchema_mappedItems.mappedItems_id)"
                + " WHERE StorageMetadata_modelId=?",
                (dataset_name,),
            )

            input_mapping = {}
            output_mapping = {}
            for original_name, mapped_name in cursor.fetchall():
                if original_name in input_names:
                    input_mapping[original_name] = mapped_name
                elif original_name in output_names:
                    output_mapping[original_name] = mapped_name

            return input_mapping, output_mapping

    def read_data_as_pandas(
        self, dataset_name: str, start_row: int = None, n_rows: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Read the input, output, and metadata of a legacy DB"""
        if start_row is None:
            start_row = 0

        if n_rows is None:
            n_rows = self.dataset_rows(dataset_name)

        with self.connection_manager as (conn, cursor):
            # get a uuid to use as a string separator in returned sql
            sep = f",-{uuid.uuid4()}-,"

            cursor.execute(
                f"SELECT timestamp, modelId, rowId, tag, GROUP_CONCAT(serializableObject SEPARATOR '{sep}')"
                + " FROM DataframeRow"
                + " LEFT JOIN (DataframeRow_Values) ON (DataframeRow.dbId = DataframeRow_Values.DataframeRow_dbId)"
                + " WHERE modelId=?"
                + " GROUP BY DataframeRow.dbId"
                + " LIMIT ? OFFSET ?;",
                (dataset_name, n_rows, start_row),
            )

            internal_metadata = []
            data = []
            for row in cursor:
                internal_metadata.append({"timestamp": row[0], "rowId": row[2], "tag": row[3]})

                parsed_values = []
                for value in row[-1].split(sep.encode()):
                    obj = javaobj.loads(value).object
                    parsed_values.append(obj.value if "value" in dir(obj) else obj)
                data.append(parsed_values)
            all_data_df = pd.DataFrame(data)
            metadata_df = pd.DataFrame(internal_metadata)

        all_data_df.columns = self._get_column_names(dataset_name)
        input_columns, output_columns = self._get_input_and_output_columns(dataset_name, all_data_df.columns)
        return all_data_df[input_columns], all_data_df[output_columns], metadata_df

    async def migrate_data(self, new_maria_storage: StorageInterface):
        """MMigrate all legacy datasets to a new storage interface"""
        existing_datasets = await new_maria_storage.list_all_datasets()
        migrations = []

        for dataset_name in self.list_datasets():
            input_dataset = dataset_name + INPUT_SUFFIX
            output_dataset = dataset_name + OUTPUT_SUFFIX
            metadata_dataset = dataset_name + METADATA_SUFFIX

            # check if migration has already happened
            input_has_migrated = input_dataset in existing_datasets
            output_has_migrated = output_dataset in existing_datasets
            metadata_has_migrated = metadata_dataset in existing_datasets

            if input_has_migrated and output_has_migrated and metadata_has_migrated:
                migrations.append(False)
                continue  # we've already migrated this DB
            else:
                input_df, output_df, metadata_df = self.read_data_as_pandas(dataset_name)
                input_mapping, output_mapping = self._get_name_mapping(
                    dataset_name, input_df.columns.values, output_df.columns.values
                )

                if not input_has_migrated:
                    await new_maria_storage.write_data(input_dataset, input_df.to_numpy(), input_df.columns.to_list())
                    await new_maria_storage.apply_name_mapping(input_dataset, input_mapping)
                if not output_has_migrated:
                    await new_maria_storage.write_data(output_dataset, output_df.to_numpy(),
                                                       output_df.columns.to_list())
                    await new_maria_storage.apply_name_mapping(output_dataset, output_mapping)
                if not metadata_has_migrated:
                    await new_maria_storage.write_data(metadata_dataset, metadata_df.to_numpy(),
                                                       metadata_df.columns.to_list())
                migrations.append(True)
                logger.info(f"Dataset {dataset_name} successfully migrated.")

        if not any(migrations):
            logger.info("All datasets already migrated- no migration necessary.")
