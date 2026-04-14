"""Model data container for managing input, output, and metadata dataframes."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.service.constants import INPUT_SUFFIX, METADATA_SUFFIX, OUTPUT_SUFFIX
from src.service.data.storage import get_global_storage_interface

logger = logging.getLogger(__name__)


@dataclass
class DataArrayConfig:
    """Configuration for a data array with column names."""

    data: np.ndarray
    names: list[str]


@dataclass
class ModelDataContainer:
    """Container for model input, output, and metadata arrays.

    Example usage:
        container = ModelDataContainer(
            model_name="my-model",
            input_config=DataArrayConfig(data=input_arr, names=["f1", "f2"]),
            output_config=DataArrayConfig(data=output_arr, names=["pred"]),
            metadata_config=DataArrayConfig(data=meta_arr, names=["id", "ts"]),
        )
    """

    model_name: str
    input_config: DataArrayConfig
    output_config: DataArrayConfig
    metadata_config: DataArrayConfig

    @property
    def input_data(self) -> np.ndarray:
        """Get input data array."""
        return self.input_config.data

    @property
    def input_names(self) -> list[str]:
        """Get input column names."""
        return self.input_config.names

    @property
    def output_data(self) -> np.ndarray:
        """Get output data array."""
        return self.output_config.data

    @property
    def output_names(self) -> list[str]:
        """Get output column names."""
        return self.output_config.names

    @property
    def metadata(self) -> np.ndarray:
        """Get metadata array."""
        return self.metadata_config.data

    @property
    def metadata_names(self) -> list[str]:
        """Get metadata column names."""
        return self.metadata_config.names


class ModelData:
    """Main interface for retrieving model data, e.g.

    model_data = ModelData("example-model-name") input_data_array,
    output_data_array, metadata_array = model_data.data()
    """

    def __init__(self, model_name: str) -> None:
        """Initialize model data accessor.

        :param model_name: Name of the model
        """
        self.model_name = model_name
        self.input_dataset = self.model_name + INPUT_SUFFIX
        self.output_dataset = self.model_name + OUTPUT_SUFFIX
        self.metadata_dataset = self.model_name + METADATA_SUFFIX

    async def datasets_exist(self) -> tuple[bool, bool, bool]:
        """Check if the requested model exists."""
        storage_interface = get_global_storage_interface()
        input_exists = await storage_interface.dataset_exists(self.input_dataset)
        output_exists = await storage_interface.dataset_exists(self.output_dataset)
        metadata_exists = await storage_interface.dataset_exists(self.metadata_dataset)

        # warn if we're missing one of the expected datasets
        dataset_checks = (input_exists, output_exists, metadata_exists)
        if not all(dataset_checks):
            expected_datasets = [
                self.input_dataset,
                self.output_dataset,
                self.metadata_dataset,
            ]
            missing_datasets = [
                dataset
                for idx, dataset in enumerate(expected_datasets)
                if not dataset_checks[idx]
            ]
            logger.warning(
                "Not all datasets present for model %s: missing %s. This could be indicative of "
                "storage corruption or improper saving of previous model data.",
                self.model_name,
                missing_datasets,
            )
        return dataset_checks

    async def row_counts(self) -> tuple[int, int, int]:
        """Get the number of input, output, and metadata rows that exist in a.

        model dataset.
        """
        storage_interface = get_global_storage_interface()
        input_rows = await storage_interface.dataset_rows(self.input_dataset)
        output_rows = await storage_interface.dataset_rows(self.output_dataset)
        metadata_rows = await storage_interface.dataset_rows(self.metadata_dataset)
        return input_rows, output_rows, metadata_rows

    async def shapes(self) -> tuple[list[int], list[int], list[int]]:
        """Get the shapes of the input, output, and metadata datasets that.

        exist in a model dataset.
        """
        storage_interface = get_global_storage_interface()
        input_shape = await storage_interface.dataset_shape(self.input_dataset)
        output_shape = await storage_interface.dataset_shape(self.output_dataset)
        metadata_shape = await storage_interface.dataset_shape(self.metadata_dataset)
        return input_shape, output_shape, metadata_shape

    async def column_names(self) -> tuple[list[str], list[str], list[str]]:
        """Get aliased column names for input, output, and metadata."""
        storage_interface = get_global_storage_interface()
        input_names = await storage_interface.get_aliased_column_names(
            self.input_dataset
        )
        output_names = await storage_interface.get_aliased_column_names(
            self.output_dataset
        )
        # these can't be aliased
        metadata_names = await storage_interface.get_original_column_names(
            self.metadata_dataset
        )

        return input_names, output_names, metadata_names

    async def original_column_names(self) -> tuple[list[str], list[str], list[str]]:
        """Get original column names for input, output, and metadata."""
        storage_interface = get_global_storage_interface()
        input_names = await storage_interface.get_original_column_names(
            self.input_dataset
        )
        output_names = await storage_interface.get_original_column_names(
            self.output_dataset
        )
        metadata_names = await storage_interface.get_original_column_names(
            self.metadata_dataset
        )

        return input_names, output_names, metadata_names

    async def data(
        self,
        start_row: int = 0,
        n_rows: int | None = None,
        *,
        get_input: bool = True,
        get_output: bool = True,
        get_metadata: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Get data from a saved model.

        * start_row: index of the first row to retreive
        * n_rows: number of rows to retrieve
        * get_input: whether to retrieve input data <- use this to reduce file reads
        * get_output: whether to retrieve output data <- use this to reduce file reads
        * get_metadata: whether to retrieve metadata <- use this to reduce file reads
        """
        storage_interface = get_global_storage_interface()

        if get_input:
            input_data = await storage_interface.read_data(
                self.input_dataset, start_row, n_rows
            )
        else:
            input_data = None
        if get_output:
            output_data = await storage_interface.read_data(
                self.output_dataset, start_row, n_rows
            )
        else:
            output_data = None
        if get_metadata:
            metadata = await storage_interface.read_data(
                self.metadata_dataset, start_row, n_rows
            )
        else:
            metadata = None

        return input_data, output_data, metadata

    async def get_metadata_as_df(self) -> pd.DataFrame:
        """Get metadata as a pandas DataFrame with validation for missing or.

        misaligned data.
        """
        _, _, metadata = await self.data(get_input=False, get_output=False)
        metadata_cols = (await self.column_names())[2]

        # Check if metadata or columns are missing
        if metadata is None or metadata_cols is None:
            logger.warning(
                "Metadata or metadata columns missing for model %s; returning empty DataFrame.",
                self.model_name,
            )
            return pd.DataFrame()

        # Check if metadata is empty
        if len(metadata) == 0 or len(metadata_cols) == 0:
            logger.warning(
                "Metadata or metadata columns empty for model %s; returning empty DataFrame.",
                self.model_name,
            )
            return pd.DataFrame()

        # Validate that metadata rows are properly formatted
        if not all(isinstance(row, (list, tuple, np.ndarray)) for row in metadata):
            logger.warning(
                "Metadata format is invalid for model %s; returning empty DataFrame.",
                self.model_name,
            )
            return pd.DataFrame()

        # Check if columns and data are aligned
        if not all(len(row) == len(metadata_cols) for row in metadata):
            logger.warning(
                "Metadata rows and columns are not aligned for model %s; returning empty DataFrame.",
                self.model_name,
            )
            return pd.DataFrame()

        return pd.DataFrame(metadata, columns=metadata_cols)

    async def summary_string(self) -> str:
        """Generate a summary string of the model data."""
        out = f"=== {self.model_name} Data ==="

        input_shape, output_shape, metadata_shape = await self.shapes()
        input_names, output_names, metadata_names = await self.column_names()

        out += f"\n\tInput Shape:    {input_shape}"
        out += f"\n\tInput Names:    {input_names}"

        out += f"\n\tOutput Shape:   {output_shape}"
        out += f"\n\tOutput Names:   {output_names}"

        out += f"\n\tMetadata Shape: {metadata_shape}"
        out += f"\n\tMetadata Names: {metadata_names}"

        return out
