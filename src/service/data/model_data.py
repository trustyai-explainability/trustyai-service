from typing import List, Optional

import numpy as np

from src.service.data.storage import get_storage_interface
from src.service.constants import *

storage_interface = get_storage_interface()


class ModelDataContainer:
    def __init__(self, model_name: str, input_data: np.ndarray, input_names: List[str], output_data: np.ndarray,
                 output_names: List[str], metadata: np.ndarray, metadata_names: List[str]):
        self.model_name = model_name
        self.input_data = input_data
        self.input_names = input_names
        self.output_data = output_data
        self.output_names = output_names
        self.metadata = metadata
        self.metadata_names = metadata_names


class ModelData:
    """Main interface for retrieving model data, e.g.

    model_data = ModelData("example-model-name")
    input_data_array, output_data_array, metadata_array = model_data.data()

    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.input_dataset = self.model_name+INPUT_SUFFIX
        self.output_dataset = self.model_name+OUTPUT_SUFFIX
        self.metadata_dataset = self.model_name+METADATA_SUFFIX

    async def row_counts(self) -> tuple[int, int, int]:
        """
        Get the number of input, output, and metadata rows that exist in a model dataset
        """
        input_rows = await storage_interface.dataset_rows(self.input_dataset)
        output_rows = await storage_interface.dataset_rows(self.output_dataset)
        metadata_rows = await storage_interface.dataset_rows(self.metadata_dataset)
        return input_rows, output_rows, metadata_rows

    async def shapes(self) -> tuple[List[int], List[int], List[int]]:
        """
        Get the shapes of the input, output, and metadata datasets that exist in a model dataset
        """
        input_shape = await storage_interface.dataset_shape(self.input_dataset)
        output_shape = await storage_interface.dataset_shape(self.output_dataset)
        metadata_shape = await storage_interface.dataset_shape(self.metadata_dataset)
        return input_shape, output_shape, metadata_shape

    async def column_names(self) -> tuple[List[str], List[str], List[str]]:
        input_names = await storage_interface.get_aliased_column_names(self.input_dataset)
        output_names = await storage_interface.get_aliased_column_names(self.output_dataset)

        # these can't be aliased
        metadata_names = await storage_interface.get_original_column_names(self.metadata_dataset)

        return input_names, output_names, metadata_names

    async def original_column_names(self) -> tuple[List[str], List[str], List[str]]:
        input_names = await storage_interface.get_original_column_names(self.input_dataset)
        output_names = await storage_interface.get_original_column_names(self.output_dataset)
        metadata_names = await storage_interface.get_original_column_names(self.metadata_dataset)

        return input_names, output_names, metadata_names

    async def data(self, start_row=None, n_rows=None, get_input=True, get_output=True, get_metadata=True) \
            -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get data from a saved model

        * start_row: index of the first row to retreive
        * n_rows: number of rows to retrieve
        * get_input: whether to retrieve input data <- use this to reduce file reads
        * get_output: whether to retrieve output data <- use this to reduce file reads
        * get_metadata: whether to retrieve metadata <- use this to reduce file reads
        """
        if get_input:
            input_data = await storage_interface.read_data(self.input_dataset, start_row, n_rows)
        else:
            input_data = None
        if get_output:
            output_data = await storage_interface.read_data(self.output_dataset, start_row, n_rows)
        else:
            output_data = None
        if get_metadata:
            metadata = await storage_interface.read_data(self.metadata_dataset, start_row, n_rows)
        else:
            metadata = None

        return input_data, output_data, metadata

    async def summary_string(self):
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
