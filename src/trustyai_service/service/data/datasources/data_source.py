"""Data source interface for loading and managing model inference data."""

import logging
import os

import pandas as pd

from trustyai_service.service.constants import (
    GROUND_TRUTH_SUFFIX,
    INTERNAL_DATA_FILENAME,
    METADATA_FILENAME,
    UNLABELED_TAG,
)
from trustyai_service.service.data.exceptions import (
    DataframeCreateError,
    StorageReadError,
)
from trustyai_service.service.data.metadata.storage_metadata import (
    StorageMetadata,
    StorageMetadataConfig,
)
from trustyai_service.service.data.model_data import ModelData
from trustyai_service.service.data.storage import get_storage_interface
from trustyai_service.service.payloads.service.schema import Schema
from trustyai_service.service.payloads.service.schema_item import SchemaItem
from trustyai_service.service.payloads.values.data_type import DataType

logger: logging.Logger = logging.getLogger(__name__)

# Array dimensionality constants for NumPy shape checks
ARRAY_DIM_2D = 2


class DataSource:
    """Manages model inference data loading and schema operations."""

    METADATA_FILENAME = METADATA_FILENAME
    GROUND_TRUTH_SUFFIX = GROUND_TRUTH_SUFFIX
    INTERNAL_DATA_FILENAME = INTERNAL_DATA_FILENAME

    def __init__(self) -> None:
        """Initialize data source with storage interface and metadata cache."""
        self.known_models: set[str] = set()
        self.storage_interface = get_storage_interface()
        self.metadata_cache: dict[str, StorageMetadata] = {}

    # MODEL TRACKING OPERATIONS

    async def get_known_models(self) -> set[str]:
        """Get the set of known model IDs."""
        return self.known_models.copy()

    async def add_model_to_known(self, model_id: str) -> None:
        """Add a model to the known models set."""
        self.known_models.add(model_id)

    # DATAFRAME READS

    async def get_dataframe(self, model_id: str) -> pd.DataFrame:
        """Get a dataframe for the given model ID using the default batch size.

        Args:
            model_id: The model ID

        Returns:
            A pandas DataFrame with the model data

        Raises:
            DataframeCreateError: If the dataframe cannot be created

        """
        batch_size = int(os.environ.get("SERVICE_BATCH_SIZE", "100"))
        return await self.get_dataframe_with_batch_size(model_id, batch_size)

    async def get_dataframe_with_batch_size(
        self, model_id: str, batch_size: int
    ) -> pd.DataFrame:
        """Get a dataframe consisting of the last `batch_size` rows of data from the corresponding model.

        Args:
            model_id: The model ID
            batch_size: The number of rows to include

        Returns:
            A pandas DataFrame with the model data

        Raises:
            DataframeCreateError: If the dataframe cannot be created

        """
        try:
            model_data = ModelData(model_id)

            input_rows, output_rows, metadata_rows = await model_data.row_counts()

            available_rows = min(input_rows, output_rows, metadata_rows)

            start_row = max(0, available_rows - batch_size)
            n_rows = min(batch_size, available_rows)

            input_data, output_data, metadata = await model_data.data(
                start_row=start_row, n_rows=n_rows
            )

            input_names, output_names, metadata_names = await model_data.column_names()

            # Combine the data into a single dataframe
            df_data = {}

            if input_data is not None:
                # Input data is expected to be 2D
                for i, col_name in enumerate(input_names):
                    if (
                        len(input_data.shape) == ARRAY_DIM_2D
                        and i < input_data.shape[1]
                    ):
                        df_data[col_name] = input_data[:, i]
                    elif len(input_data.shape) == 1 and i == 0:
                        # Single column case (1D array)
                        df_data[col_name] = input_data

            if output_data is not None:
                # Output data can be 1D (single output) or 2D (multiple outputs)
                if len(output_data.shape) == 1:
                    # 1D array - single output column
                    if len(output_names) > 0:
                        df_data[output_names[0]] = output_data
                else:
                    # 2D array - multiple output columns
                    for i, col_name in enumerate(output_names):
                        if i < output_data.shape[1]:
                            df_data[col_name] = output_data[:, i]

            if metadata is not None:
                # Metadata can be 1D (single column) or 2D (multiple columns)
                if len(metadata.shape) == 1:
                    # 1D array - single metadata column
                    if len(metadata_names) > 0:
                        df_data[metadata_names[0]] = metadata
                else:
                    # 2D array - multiple metadata columns
                    for i, col_name in enumerate(metadata_names):
                        if i < metadata.shape[1]:
                            df_data[col_name] = metadata[:, i]

            return pd.DataFrame(df_data)

        except Exception as e:  # Broad catch intentional: dataframe creation involves dynamic storage operations
            logger.exception("Error creating dataframe for model=%s", model_id)
            msg = f"Error creating dataframe for model={model_id}: {e!s}"
            raise DataframeCreateError(msg) from e

    async def get_organic_dataframe(
        self, model_id: str, batch_size: int
    ) -> pd.DataFrame:
        """Get a dataframe with only organic data (not synthetic).

        Args:
            model_id: The model ID
            batch_size: The number of rows to include

        Returns:
            A pandas DataFrame with organic model data

        Raises:
            DataframeCreateError: If the dataframe cannot be created

        """
        df = await self.get_dataframe_with_batch_size(model_id, batch_size)

        # Filter out any rows with the unlabeled tag (synthetic data)
        if UNLABELED_TAG in df.columns:
            df = df[~df[UNLABELED_TAG].fillna(value=False)]

        return df

    # METADATA READS

    async def get_metadata(self, model_id: str) -> StorageMetadata:
        """Get metadata for the given model ID.

        Args:
            model_id: The model ID

        Returns:
            A StorageMetadata object

        Raises:
            StorageReadError: If the metadata cannot be retrieved

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
                StorageMetadataConfig(
                    model_id=model_id,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    observations=min(input_rows, output_rows, metadata_rows),
                    recorded_inferences=UNLABELED_TAG in metadata_names,
                )
            )

            self.metadata_cache[model_id] = metadata

        except Exception as e:  # Broad catch intentional: metadata retrieval involves dynamic storage operations
            logger.exception("Error getting metadata for model=%s", model_id)
            msg = f"Error getting metadata for model={model_id}: {e!s}"
            raise StorageReadError(msg) from e
        else:
            return metadata

    async def has_metadata(self, model_id: str) -> bool:
        """Check if metadata exists for the given model ID.

        Args:
            model_id: The model ID

        Returns:
            True if metadata exists, False otherwise

        """
        try:
            return await self.get_metadata(model_id) is not None
        except Exception:  # Broad catch intentional: metadata check errors should return False, not crash
            logger.exception("Error checking if metadata exists for model=%s", model_id)
            return False

    # DATAFRAME QUERIES

    async def get_num_observations(self, model_id: str) -> int:
        """Get the number of observations for the corresponding model.

        Args:
            model_id: The model ID

        Returns:
            The number of observations

        """
        metadata: StorageMetadata = await self.get_metadata(model_id)
        return metadata.get_observations()

    async def has_recorded_inferences(self, model_id: str) -> bool:
        """Check to see if a particular model has recorded inferences.

        Args:
            model_id: The model ID

        Returns:
            True if the model has received inference data

        """
        metadata: StorageMetadata = await self.get_metadata(model_id)
        return metadata.is_recorded_inferences()

    async def get_verified_models(self) -> list[str]:
        """Get the list of model IDs that are confirmed to have metadata in storage.

        Returns:
            A list of verified model IDs

        """
        # Snapshot: iteration awaits, so the set could change at yield points
        verified_models = [
            model_id
            for model_id in list(self.known_models)
            if await self.has_metadata(model_id)
        ]

        if not verified_models:
            discovered_models = await self._discover_models_from_storage()
            # Cannot use list comprehension due to side effects (add_model_to_known)
            for model_id in discovered_models:
                if await self.has_metadata(model_id):
                    await self.add_model_to_known(model_id)
                    verified_models.append(model_id)

        return verified_models

    async def _discover_models_from_storage(self) -> list[str]:
        """Discover model IDs from storage by scanning for actual model data.

        Returns:
            A list of discovered model IDs

        """
        discovered = []

        try:
            # Use storage interface to discover models with actual data
            storage_models = await self.storage_interface.get_known_models()
            logger.info("Storage interface discovered models: %s", storage_models)
            discovered.extend(storage_models)

        except (
            Exception
        ) as e:  # Intentional: storage failures should not break model discovery
            logger.warning("Failed to discover models from storage interface: %s", e)

        logger.info("Total discovered models: %s", discovered)
        return discovered

    # GROUND TRUTH OPERATIONS

    @staticmethod
    def get_ground_truth_name(model_id: str) -> str:
        """Get the ground truth name for a model.

        Args:
            model_id: The model ID

        Returns:
            The ground truth name

        """
        return model_id + DataSource.GROUND_TRUTH_SUFFIX

    async def has_ground_truths(self, model_id: str) -> bool:
        """Check if ground truths exist for a model.

        Args:
            model_id: The model ID

        Returns:
            True if ground truths exist, False otherwise

        """
        return await self.has_metadata(self.get_ground_truth_name(model_id))

    async def get_ground_truths(self, model_id: str) -> pd.DataFrame:
        """Get ground-truth dataframe for this particular model.

        Args:
            model_id: The model ID for which these ground truths apply

        Returns:
            The ground-truth dataframe

        """
        return await self.get_dataframe(self.get_ground_truth_name(model_id))

    # UTILITY METHODS

    async def save_dataframe(
        self, _dataframe: pd.DataFrame, model_id: str, *, overwrite: bool = False
    ) -> None:
        """Save a dataframe for the given model ID.

        Args:
            _dataframe: The dataframe to save
            model_id: The model ID
            overwrite: If true, overwrite existing data. Otherwise, append.

        """
        # Add to known models
        await self.add_model_to_known(model_id)

        # No-op: testing implementation does not persist to storage
        logger.info("Saving dataframe for model %s (overwrite=%s)", model_id, overwrite)

    def save_metadata(self, storage_metadata: StorageMetadata, model_id: str) -> None:
        """Save metadata for this model ID.

        Args:
            storage_metadata: The metadata to save
            model_id: The model ID to save this metadata under

        """
        # Update cache
        self.metadata_cache[model_id] = storage_metadata

        # No-op: testing implementation does not persist to storage
        logger.info("Saving metadata for model %s", model_id)
