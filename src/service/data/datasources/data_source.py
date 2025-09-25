import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set

import pandas as pd
from src.service.constants import (
    GROUND_TRUTH_SUFFIX,
    INTERNAL_DATA_FILENAME,
    METADATA_FILENAME,
    UNLABELED_TAG,
)
from src.service.data.exceptions import DataframeCreateException, StorageReadException
from src.service.data.metadata.storage_metadata import StorageMetadata

from src.service.data.model_data import ModelData
from src.service.data.storage import get_storage_interface
from src.service.payloads.service.schema import Schema
from src.service.payloads.service.schema_item import SchemaItem
from src.service.payloads.values.data_type import DataType

logger: logging.Logger = logging.getLogger(__name__)


# TODO: This class is a placeholder for the actual data source implementation.
# It is here to provide a place to put the data source implementation when it is ready.
class DataSource:
    METADATA_FILENAME = METADATA_FILENAME
    GROUND_TRUTH_SUFFIX = GROUND_TRUTH_SUFFIX
    INTERNAL_DATA_FILENAME = INTERNAL_DATA_FILENAME

    def __init__(self) -> None:
        self.known_models: Set[str] = set()
        self.storage_interface = get_storage_interface()
        self.metadata_cache: Dict[str, StorageMetadata] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)

    # MODEL TRACKING OPERATIONS

    async def get_known_models(self) -> Set[str]:
        """Get the set of known model IDs."""
        return self.known_models.copy()

    async def add_model_to_known(self, model_id: str) -> None:
        """Add a model to the known models set."""
        self.known_models.add(model_id)

    # DATAFRAME READS

    async def get_dataframe(self, model_id: str) -> pd.DataFrame:
        """
        Get a dataframe for the given model ID using the default batch size.

        Args:
            model_id: The model ID

        Returns:
            A pandas DataFrame with the model data

        Raises:
            DataframeCreateException: If the dataframe cannot be created
        """
        batch_size = int(os.environ.get("SERVICE_BATCH_SIZE", "100"))
        return await self.get_dataframe_with_batch_size(model_id, batch_size)

    async def get_dataframe_with_batch_size(self, model_id: str, batch_size: int) -> pd.DataFrame:
        """
        Get a dataframe consisting of the last `batch_size` rows of data from the corresponding model.

        Args:
            model_id: The model ID
            batch_size: The number of rows to include

        Returns:
            A pandas DataFrame with the model data

        Raises:
            DataframeCreateException: If the dataframe cannot be created
        """
        try:
            model_data = ModelData(model_id)

            input_rows, output_rows, metadata_rows = await model_data.row_counts()

            available_rows = min(input_rows, output_rows, metadata_rows)

            start_row = max(0, available_rows - batch_size)
            n_rows = min(batch_size, available_rows)

            input_data, output_data, metadata = await model_data.data(start_row=start_row, n_rows=n_rows)

            input_names, output_names, metadata_names = await model_data.column_names()

            # Combine the data into a single dataframe
            df_data = {}

            if input_data is not None:
                for i, col_name in enumerate(input_names):
                    if i < input_data.shape[1]:
                        df_data[col_name] = input_data[:, i]

            if output_data is not None:
                for i, col_name in enumerate(output_names):
                    if i < output_data.shape[1]:
                        df_data[col_name] = output_data[:, i]

            if metadata is not None:
                for i, col_name in enumerate(metadata_names):
                    if i < metadata.shape[1]:
                        df_data[col_name] = metadata[:, i]

            return pd.DataFrame(df_data)

        except Exception as e:
            logger.error(f"Error creating dataframe for model={model_id}: {str(e)}")
            raise DataframeCreateException(f"Error creating dataframe for model={model_id}: {str(e)}")

    async def get_organic_dataframe(self, model_id: str, batch_size: int) -> pd.DataFrame:
        """
        Get a dataframe with only organic data (not synthetic).

        Args:
            model_id: The model ID
            batch_size: The number of rows to include

        Returns:
            A pandas DataFrame with organic model data

        Raises:
            DataframeCreateException: If the dataframe cannot be created
        """
        df = await self.get_dataframe_with_batch_size(model_id, batch_size)

        # Filter out any rows with the unlabeled tag (synthetic data)
        if UNLABELED_TAG in df.columns:
            df = df[df[UNLABELED_TAG] != True]

        return df

    # METADATA READS

    async def get_metadata(self, model_id: str) -> StorageMetadata:
        """
        Get metadata for the given model ID.

        Args:
            model_id: The model ID

        Returns:
            A StorageMetadata object

        Raises:
            StorageReadException: If the metadata cannot be retrieved
        """
        if model_id in self.metadata_cache:
            return self.metadata_cache[model_id]

        try:
            model_data = ModelData(model_id)

            input_rows, output_rows, metadata_rows = await model_data.row_counts()
            input_names, output_names, metadata_names = await model_data.column_names()

            input_items = {}
            for i, name in enumerate(input_names):
                # Default to STRING type - in a real implementation this would be determined from the data
                input_items[name] = SchemaItem(DataType.STRING, name, i)
            input_schema = Schema(input_items)

            output_items = {}
            for i, name in enumerate(output_names):
                output_items[name] = SchemaItem(DataType.STRING, name, i)
            output_schema = Schema(output_items)

            metadata = StorageMetadata(
                model_id=model_id,
                input_schema=input_schema,
                output_schema=output_schema,
                observations=min(input_rows, output_rows, metadata_rows),
                recorded_inferences=UNLABELED_TAG in metadata_names,
            )

            self.metadata_cache[model_id] = metadata
            return metadata

        except Exception as e:
            logger.error(f"Error getting metadata for model={model_id}: {str(e)}")
            raise StorageReadException(f"Error getting metadata for model={model_id}: {str(e)}")

    async def has_metadata(self, model_id: str) -> bool:
        """
        Check if metadata exists for the given model ID.

        Args:
            model_id: The model ID

        Returns:
            True if metadata exists, False otherwise
        """
        try:
            return await self.get_metadata(model_id) is not None
        except Exception as e:
            logger.error(f"Error checking if metadata exists for model={model_id}: {str(e)}")
            return False

    # DATAFRAME QUERIES

    async def get_num_observations(self, model_id: str) -> int:
        """
        Get the number of observations for the corresponding model.

        Args:
            model_id: The model ID

        Returns:
            The number of observations
        """
        metadata: StorageMetadata = await self.get_metadata(model_id)
        return metadata.get_observations()

    async def has_recorded_inferences(self, model_id: str) -> bool:
        """
        Check to see if a particular model has recorded inferences.

        Args:
            model_id: The model ID

        Returns:
            True if the model has received inference data
        """
        metadata: StorageMetadata = await self.get_metadata(model_id)
        return metadata.is_recorded_inferences()

    async def get_verified_models(self) -> List[str]:
        """
        Get the list of model IDs that are confirmed to have metadata in storage.

        Returns:
            A list of verified model IDs
        """
        verified_models = []

        # Check all known models for metadata
        for model_id in self.known_models:
            if await self.has_metadata(model_id):
                verified_models.append(model_id)

        if not verified_models:
            discovered_models = await self._discover_models_from_storage()
            for model_id in discovered_models:
                if await self.has_metadata(model_id):
                    await self.add_model_to_known(model_id)
                    verified_models.append(model_id)

        return verified_models

    async def _discover_models_from_storage(self) -> List[str]:
        """
        Discover model IDs from storage.

        Returns:
            A list of discovered model IDs
        """
        # TODO: In a real implementation, this would scan the storage for model directories
        # For now, return any models that might be in environment or configuration
        discovered = []

        # Check for models mentioned in environment variables
        if "TEST_MODEL_ID" in os.environ:
            discovered.append(os.environ["TEST_MODEL_ID"])

        return discovered

    # GROUND TRUTH OPERATIONS

    @staticmethod
    def get_ground_truth_name(model_id: str) -> str:
        """
        Get the ground truth name for a model.

        Args:
            model_id: The model ID

        Returns:
            The ground truth name
        """
        return model_id + DataSource.GROUND_TRUTH_SUFFIX

    async def has_ground_truths(self, model_id: str) -> bool:
        """
        Check if ground truths exist for a model.

        Args:
            model_id: The model ID

        Returns:
            True if ground truths exist, False otherwise
        """
        return await self.has_metadata(self.get_ground_truth_name(model_id))

    async def get_ground_truths(self, model_id: str) -> pd.DataFrame:
        """
        Get ground-truth dataframe for this particular model.

        Args:
            model_id: The model ID for which these ground truths apply

        Returns:
            The ground-truth dataframe
        """
        return await self.get_dataframe(self.get_ground_truth_name(model_id))

    # UTILITY METHODS

    async def save_dataframe(self, dataframe: pd.DataFrame, model_id: str, overwrite: bool = False) -> None:
        """
        Save a dataframe for the given model ID.

        Args:
            dataframe: The dataframe to save
            model_id: The model ID
            overwrite: If true, overwrite existing data. Otherwise, append.
        """
        # Add to known models
        await self.add_model_to_known(model_id)

        # TODO: In a full implementation, this would save the dataframe to storage
        logger.info(f"Saving dataframe for model {model_id} (overwrite={overwrite})")

    def save_metadata(self, storage_metadata: StorageMetadata, model_id: str) -> None:
        """
        Save metadata for this model ID.

        Args:
            storage_metadata: The metadata to save
            model_id: The model ID to save this metadata under
        """
        # Update cache
        self.metadata_cache[model_id] = storage_metadata

        # TODO: In a full implementation, this would save to storage
        logger.info(f"Saving metadata for model {model_id}")
