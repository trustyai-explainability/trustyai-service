import asyncio
from typing import List, Dict, Optional

import numpy as np
import os
import h5py
import logging
import pickle as pkl

from src.service.utils import list_utils
from .storage_interface import StorageInterface
from src.service.constants import PROTECTED_DATASET_SUFFIX, PARTIAL_PAYLOAD_DATASET_NAME
from src.service.data.modelmesh_parser import PartialPayload
from src.service.data.metadata.storage_metadata import StorageMetadata
from src.service.payloads.service.schema import Schema
from src.service.payloads.service.schema_item import SchemaItem
from src.service.payloads.values.data_type import DataType

logger = logging.getLogger(__name__)
COLUMN_NAMES_ATTRIBUTE = "column_names"
COLUMN_ALIAS_ATTRIBUTE = "column_aliases"
BYTES_ATTRIBUTE = "is_bytes"

PARTIAL_INPUT_NAME = PROTECTED_DATASET_SUFFIX + PARTIAL_PAYLOAD_DATASET_NAME + "_inputs"
PARTIAL_OUTPUT_NAME = PROTECTED_DATASET_SUFFIX + PARTIAL_PAYLOAD_DATASET_NAME + "_outputs"
MODELMESH_INPUT_NAME = f"{PROTECTED_DATASET_SUFFIX}modelmesh_partial_payloads_inputs"
MODELMESH_OUTPUT_NAME = f"{PROTECTED_DATASET_SUFFIX}modelmesh_partial_payloads_outputs"


class H5PYContext:
    """Open the corresponding H5PY file for a dataset and manage its context`"""

    def __init__(self, parent_class, dataset_name, mode):
        self.parent_class = parent_class
        self.mode = mode
        self.dataset_name = dataset_name
        self.filename = parent_class._get_filename(self.dataset_name)

    def __enter__(self):
        if self.mode == "r" and not os.path.exists(self.filename):
            raise MissingH5PYDataException(self.dataset_name)
        self.db = h5py.File(self.filename, mode=self.mode)
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()


class MissingH5PYDataException(Exception):
    """Raised when a dataset that does not exist is accessed"""

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __str__(self):
        return f"No inference data for dataset={self.dataset_name} found."


class PVCStorage(StorageInterface):
    def __init__(self, data_directory, data_file="trustyai_inference_data.hdf5"):
        self.data_path = os.path.join(data_directory, data_file)
        self.data_directory = data_directory
        self.data_file = data_file

        # save all datasets into the same H5PY or one file per dataset?
        # one_file_per_dataset=True minimizes the ramifications of file corruption
        # one_file_per_dataset=True also allows read/write concurrence between different datasets
        self.one_file_per_dataset = True
        self.locks = {fname: asyncio.Lock() for fname in os.listdir(self.data_directory) if self.data_file in fname}
        self.global_lock = asyncio.Lock()

    def _get_filename(self, dataset_name):
        """Get the H5PY filename of a particular dataset"""
        if self.one_file_per_dataset:
            return os.path.join(self.data_directory, f"{dataset_name}_{self.data_file}")
        else:
            return self.data_path

    def get_lock(self, dataset_name):
        """Get the per-document lock to prevent simultaneous read/writes on an individual H5PY file"""

        filename = self._get_filename(dataset_name)
        if filename not in self.locks:
            self.locks[filename] = asyncio.Lock()
        return self.locks[filename]

    @staticmethod
    def allocate_valid_dataset_name(dataset_name: str):
        """Given an arbitrary model name inbound from the model server, ensure that it does not
        conflict with any internal TrustyAI dataset names"""

        if dataset_name.startswith(PROTECTED_DATASET_SUFFIX):
            return dataset_name.replace(PROTECTED_DATASET_SUFFIX, "inference_")
        else:
            return dataset_name

    async def dataset_exists(self, dataset_name: str) -> bool:
        """Does a file exist in the data directory for this dataset?"""

        async with self.get_lock(dataset_name):
            try:
                with H5PYContext(self, dataset_name, "r") as db:
                    return dataset_name in db
            except MissingH5PYDataException:
                return False

    def list_all_datasets(self) -> List[str]:
        """List all datasets known by the dataset"""
        return [
            fname.replace(f"_{self.data_file}", "")
            for fname in os.listdir(self.data_directory)
            if self.data_file in fname
        ]

    async def dataset_rows(self, dataset_name: str) -> int:
        """Number of data rows in dataset, returns a FileNotFoundError if the dataset does not exist"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    return db[allocated_dataset_name].shape[0]
                else:
                    raise MissingH5PYDataException(allocated_dataset_name)

    async def dataset_shape(self, dataset_name: str) -> tuple[int]:
        """Shape of the dataset, returns a FileNotFoundError if the dataset does not exist"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    return db[allocated_dataset_name].shape
                else:
                    raise MissingH5PYDataException(allocated_dataset_name)

    async def _write_raw_data(
        self,
        dataset_name: str,
        new_rows: np.ndarray,
        column_names: list[str],
        is_bytes: bool = False,
    ) -> None:
        """Write new data to file. Axis 0 of the data is the row dimension, and data shape must
        align on all subsequent axes"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        inbound_shape = list(new_rows.shape)

        # use the dataset_shape function to check both existence of dataset + shape retrieval, to reduce file reads
        try:
            existing_shape = list(await self.dataset_shape(allocated_dataset_name))
            dataset_exists = True
        except MissingH5PYDataException:
            existing_shape = None
            dataset_exists = False

        if dataset_exists:  # if we've already got saved inferences for this model
            if existing_shape[1:] == inbound_shape[1:]:  # shapes match
                async with self.get_lock(allocated_dataset_name):
                    with H5PYContext(self, allocated_dataset_name, "a") as db:
                        dataset = db[allocated_dataset_name]

                        if dataset.attrs[BYTES_ATTRIBUTE] != is_bytes:  # data storage paradigm mismatch
                            msg = f"Error when saving inference data for {allocated_dataset_name}: "
                            if dataset.attrs[BYTES_ATTRIBUTE]:
                                msg += (
                                    "Dataset was previously saved as serialized tabular data, but has "
                                    "now received a purely numeric payload."
                                )
                            else:
                                msg += (
                                    "Dataset was previously saved as numeric data, but has now received "
                                    "a serialized tabular payload."
                                )
                            logger.error(msg)
                            raise ValueError(msg)

                        # add new lines to dataset and write new data
                        dataset.resize(existing_shape[0] + inbound_shape[0], axis=0)
                        dataset[existing_shape[0] :] = new_rows
            else:
                existing_shape_str = ", ".join([":"] + [str(x) for x in existing_shape[1:]])
                inbound_shape_str = ", ".join([":"] + [str(x) for x in inbound_shape[1:]])

                raise ValueError(
                    f"Error when saving inference data for {allocated_dataset_name}: "
                    f"Mismatch between existing data shape=({existing_shape_str}) vs "
                    f"inbound data shape=({inbound_shape_str})"
                )
        else:  # first observation of inferences from this model
            async with self.get_lock(allocated_dataset_name):
                with H5PYContext(self, allocated_dataset_name, "a") as db:
                    # create new dataset
                    max_shape = [None] + list(new_rows.shape)[1:]  # to-do: tune this value?
                    dataset = db.create_dataset(
                        allocated_dataset_name,
                        data=new_rows,
                        maxshape=max_shape,
                        chunks=True,
                    )
                    dataset.attrs[COLUMN_NAMES_ATTRIBUTE] = column_names
                    dataset.attrs[BYTES_ATTRIBUTE] = is_bytes

    async def write_data(self, dataset_name: str, new_rows, column_names: List[str]):
        """Write new data to a dataset, automatically serializing any non-numeric data"""
        if isinstance(new_rows, np.ndarray) and not list_utils.contains_non_numeric(new_rows):
            await self._write_raw_data(dataset_name, new_rows, column_names)
        elif (
            isinstance(new_rows, np.ndarray)
            and list_utils.contains_non_numeric(new_rows)
            or not isinstance(new_rows, np.ndarray)
            and list_utils.contains_non_numeric(new_rows)
        ):
            await self._write_raw_data(dataset_name, list_utils.serialize_rows(new_rows), column_names)
        else:
            await self._write_raw_data(dataset_name, np.array(new_rows), column_names)

    async def _read_raw_data(
        self, dataset_name: str, start_row: int = None, n_rows: int = None
    ) -> (np.ndarray, List[str]):
        """Read raw data from a dataset- does not deserialize any bytes data"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name not in db:
                    raise MissingH5PYDataException(allocated_dataset_name)
                start_row = 0 if start_row is None else start_row
                end_row = None if n_rows is None else start_row + n_rows
                dataset = db[allocated_dataset_name]
                if start_row > dataset.shape[0]:
                    logger.warning(
                        f"Requested a data read from start_row={start_row}, but dataset "
                        f"only has {dataset.shape[0]} rows. An empty array will be returned."
                    )
                return (
                    dataset[start_row:end_row],
                    dataset.attrs[COLUMN_NAMES_ATTRIBUTE],
                )

    async def read_data(self, dataset_name: str, start_row: int = None, n_rows: int = None) -> (np.ndarray, List[str]):
        """Read data from a dataset, automatically deserializing any byte data"""
        read, column_names = await self._read_raw_data(dataset_name, start_row, n_rows)
        if len(read) and read[0].dtype.type in {np.bytes_, np.void}:
            return list_utils.deserialize_rows(read), column_names
        else:
            return read, column_names

    async def delete_dataset(self, dataset_name: str):
        """Delete dataset data, ignoring non-existent datasets"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            try:
                with H5PYContext(self, dataset_name, "a") as db:
                    if allocated_dataset_name in db:
                        del db[allocated_dataset_name]
                    if allocated_dataset_name in self.locks:
                        del self.locks[allocated_dataset_name]
            except MissingH5PYDataException:
                pass

    async def get_original_column_names(self, dataset_name: str) -> List[str]:
        """Get the original column names associated with this model, prior to any name mapping"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    return db[allocated_dataset_name].attrs[COLUMN_NAMES_ATTRIBUTE]
                else:
                    raise MissingH5PYDataException(allocated_dataset_name)

    async def get_aliased_column_names(self, dataset_name: str) -> List[str]:
        """Get an up-to-date set of column names, including any aliases that might have been applied"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    if COLUMN_ALIAS_ATTRIBUTE in db[dataset_name].attrs:
                        return db[allocated_dataset_name].attrs[COLUMN_ALIAS_ATTRIBUTE]
                    else:
                        return db[allocated_dataset_name].attrs[COLUMN_NAMES_ATTRIBUTE]
                else:
                    raise MissingH5PYDataException(allocated_dataset_name)

    async def apply_name_mapping(self, dataset_name: str, name_mapping: Dict[str, str]):
        """Apply a new name mapping to a dataset"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "a") as db:
                curr_names = db[allocated_dataset_name].attrs[COLUMN_NAMES_ATTRIBUTE]
                aliased_names = [name_mapping.get(name, name) for name in curr_names]
                db[allocated_dataset_name].attrs[COLUMN_ALIAS_ATTRIBUTE] = aliased_names

    async def clear_name_mapping(self, dataset_name: str):
        """Clear/remove the name mapping for a dataset"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "a") as db:
                if allocated_dataset_name in db:
                    if COLUMN_ALIAS_ATTRIBUTE in db[allocated_dataset_name].attrs:
                        del db[allocated_dataset_name].attrs[COLUMN_ALIAS_ATTRIBUTE]
                        logger.info(f"Successfully cleared name mapping for dataset '{allocated_dataset_name}'")
                    else:
                        logger.warning(
                            f"Attempted to clear name mapping for dataset '{allocated_dataset_name}', "
                            f"but '{COLUMN_ALIAS_ATTRIBUTE}' attribute was not found."
                        )
                else:
                    logger.warning(
                        f"Attempted to clear name mapping for dataset '{allocated_dataset_name}', "
                        f"but dataset was not found in the database."
                    )

    def get_known_models(self) -> List[str]:
        """Get a list of all model IDs that have inference data stored"""
        from src.service.constants import INPUT_SUFFIX, OUTPUT_SUFFIX, METADATA_SUFFIX

        all_datasets = self.list_all_datasets()
        logger.info(f"All datasets found: {all_datasets}")
        model_ids = set()

        for dataset_name in all_datasets:
            logger.debug(f"Processing dataset: {dataset_name}")
            # Skip internal datasets
            if dataset_name.startswith(PROTECTED_DATASET_SUFFIX):
                logger.debug(f"Skipping internal dataset: {dataset_name}")
                continue

            # Extract model ID by removing suffixes
            if dataset_name.endswith(INPUT_SUFFIX):
                model_id = dataset_name[:-len(INPUT_SUFFIX)]
                model_ids.add(model_id)
                logger.debug(f"Found input dataset for model: {model_id}")
            elif dataset_name.endswith(OUTPUT_SUFFIX):
                model_id = dataset_name[:-len(OUTPUT_SUFFIX)]
                model_ids.add(model_id)
                logger.debug(f"Found output dataset for model: {model_id}")
            elif dataset_name.endswith(METADATA_SUFFIX):
                model_id = dataset_name[:-len(METADATA_SUFFIX)]
                model_ids.add(model_id)
                logger.debug(f"Found metadata dataset for model: {model_id}")
            else:
                logger.debug(f"Dataset doesn't match expected suffixes: {dataset_name}")

        logger.info(f"Extracted model IDs: {list(model_ids)}")
        return list(model_ids)

    async def get_metadata(self, model_id: str) -> StorageMetadata:
        """Get metadata for a specific model including shapes, column names, etc."""
        from src.service.constants import INPUT_SUFFIX, OUTPUT_SUFFIX, METADATA_SUFFIX

        input_dataset = model_id + INPUT_SUFFIX
        output_dataset = model_id + OUTPUT_SUFFIX
        metadata_dataset = model_id + METADATA_SUFFIX

        logger.info(f"Getting metadata for model_id: {model_id}")
        logger.info(f"Looking for datasets: input={input_dataset}, output={output_dataset}, metadata={metadata_dataset}")

        # Check which datasets exist
        input_exists = await self.dataset_exists(input_dataset)
        output_exists = await self.dataset_exists(output_dataset)
        metadata_exists = await self.dataset_exists(metadata_dataset)

        logger.info(f"Dataset existence: input={input_exists}, output={output_exists}, metadata={metadata_exists}")

        metadata = {
            "modelId": model_id,
            "inputData": None,
            "outputData": None,
            "metadataData": None
        }

        # Get input data metadata
        if input_exists:
            try:
                logger.info(f"Retrieving input metadata for {input_dataset}")
                input_shape = await self.dataset_shape(input_dataset)
                input_names = await self.get_original_column_names(input_dataset)
                aliased_input_names = await self.get_aliased_column_names(input_dataset)
                logger.info(f"Input metadata: shape={input_shape}, names={input_names}, aliases={aliased_input_names}")
                metadata["inputData"] = {
                    "shape": list(input_shape) if input_shape is not None else [],
                    "columnNames": list(input_names) if input_names is not None else [],
                    "aliasedNames": list(aliased_input_names) if aliased_input_names is not None else []
                }
            except Exception as e:
                logger.error(f"Error getting input metadata for {model_id}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

        # Get output data metadata
        if output_exists:
            try:
                logger.info(f"Retrieving output metadata for {output_dataset}")
                output_shape = await self.dataset_shape(output_dataset)
                output_names = await self.get_original_column_names(output_dataset)
                aliased_output_names = await self.get_aliased_column_names(output_dataset)
                metadata["outputData"] = {
                    "shape": list(output_shape) if output_shape is not None else [],
                    "columnNames": list(output_names) if output_names is not None else [],
                    "aliasedNames": list(aliased_output_names) if aliased_output_names is not None else []
                }
            except Exception as e:
                logger.error(f"Error getting output metadata for {model_id}: {e}")

        # Get metadata data info
        if metadata_exists:
            try:
                logger.info(f"Retrieving metadata info for {metadata_dataset}")
                metadata_shape = await self.dataset_shape(metadata_dataset)
                metadata_names = await self.get_original_column_names(metadata_dataset)
                metadata["metadataData"] = {
                    "shape": list(metadata_shape) if metadata_shape is not None else [],
                    "columnNames": list(metadata_names) if metadata_names is not None else []
                }
            except Exception as e:
                logger.error(f"Error getting metadata info for {model_id}: {e}")

        # Create schemas for input and output data
        input_schema = None
        output_schema = None
        observations = 0

        try:
            # Create input schema if input data exists
            if input_exists and metadata.get("inputData"):
                input_data = metadata["inputData"]
                column_names = input_data.get("columnNames", [])
                aliased_names = input_data.get("aliasedNames", [])

                # Create name mapping from original to aliased names
                name_mapping = {}
                if len(column_names) == len(aliased_names):
                    name_mapping = {orig: alias for orig, alias in zip(column_names, aliased_names) if orig != alias}

                # Create SchemaItem objects for each column
                schema_items = {}
                for idx, col_name in enumerate(column_names):
                    schema_items[col_name] = SchemaItem(
                        type=DataType.UNKNOWN,  # We don't have type info in PVC storage
                        name=col_name,
                        column_index=idx
                    )

                input_schema = Schema(
                    items=schema_items,
                    name_mapping=name_mapping
                )

                # Get observation count from input dataset
                if input_data.get("shape"):
                    observations = input_data["shape"][0] if len(input_data["shape"]) > 0 else 0

            # Create output schema if output data exists
            if output_exists and metadata.get("outputData"):
                output_data = metadata["outputData"]
                column_names = output_data.get("columnNames", [])
                aliased_names = output_data.get("aliasedNames", [])

                # Create name mapping from original to aliased names
                name_mapping = {}
                if len(column_names) == len(aliased_names):
                    name_mapping = {orig: alias for orig, alias in zip(column_names, aliased_names) if orig != alias}

                # Create SchemaItem objects for each column
                schema_items = {}
                for idx, col_name in enumerate(column_names):
                    schema_items[col_name] = SchemaItem(
                        type=DataType.UNKNOWN,  # We don't have type info in PVC storage
                        name=col_name,
                        column_index=idx
                    )

                output_schema = Schema(
                    items=schema_items,
                    name_mapping=name_mapping
                )

            # Use default empty schemas if none exist
            if input_schema is None:
                input_schema = Schema(name_mapping={}, items={})
            if output_schema is None:
                output_schema = Schema(name_mapping={}, items={})

            # Create and return StorageMetadata object
            storage_metadata = StorageMetadata(
                model_id=model_id,
                input_schema=input_schema,
                output_schema=output_schema,
                input_tensor_name="input",  # Default tensor name
                output_tensor_name="output",  # Default tensor name
                observations=observations,
                recorded_inferences=observations > 0  # True if we have data
            )

            logger.info(f"Created StorageMetadata for {model_id}: observations={observations}, recorded_inferences={observations > 0}")
            return storage_metadata

        except Exception as e:
            logger.error(f"Error creating StorageMetadata for {model_id}: {e}")
            # Return minimal metadata object on error
            return StorageMetadata(
                model_id=model_id,
                input_schema=Schema(name_mapping={}, items={}),
                output_schema=Schema(name_mapping={}, items={}),
                input_tensor_name="input",
                output_tensor_name="output",
                observations=0,
                recorded_inferences=False
            )

    async def persist_partial_payload(self, payload, is_input: bool):
        """Save a partial payload to disk. Returns None if no matching id exists"""

        # lock to prevent simultaneous read/writes
        partial_dataset_name = PARTIAL_INPUT_NAME if is_input else PARTIAL_OUTPUT_NAME
        async with self.get_lock(partial_dataset_name):
            with H5PYContext(
                self,
                partial_dataset_name,
                "a",
            ) as db:
                if partial_dataset_name not in db:
                    dataset = db.create_dataset(partial_dataset_name, dtype="f", track_order=True)
                else:
                    dataset = db[partial_dataset_name]
                dataset.attrs[payload.id] = np.void(pkl.dumps(payload))

    async def persist_modelmesh_payload(self, payload: PartialPayload, request_id: str, is_input: bool):
        """
        Persist a ModelMesh payload.

        Args:
            payload: The payload to persist
            request_id: The unique identifier for the inference request
            is_input: Whether this is an input payload (True) or output payload (False)
        """
        dataset_name = MODELMESH_INPUT_NAME if is_input else MODELMESH_OUTPUT_NAME

        serialized_data = pkl.dumps(payload.model_dump())

        async with self.get_lock(dataset_name):
            try:
                with H5PYContext(self, dataset_name, "a") as db:
                    if dataset_name not in db:
                        dataset = db.create_dataset(dataset_name, data=np.array([0]))
                        dataset.attrs["request_ids"] = []

                    dataset = db[dataset_name]
                    request_ids = list(dataset.attrs["request_ids"])

                    dataset.attrs[request_id] = np.void(serialized_data)

                    if request_id not in request_ids:
                        request_ids.append(request_id)
                        dataset.attrs["request_ids"] = request_ids

                logger.debug(
                    f"Stored ModelMesh {'input' if is_input else 'output'} payload for request ID: {request_id}"
                )
            except Exception as e:
                logger.error(f"Error storing ModelMesh payload: {str(e)}")
                raise

    async def get_modelmesh_payload(self, request_id: str, is_input: bool) -> Optional[PartialPayload]:
        """
        Retrieve a stored ModelMesh payload by request ID.

        Args:
            request_id: The unique identifier for the inference request
            is_input: Whether to retrieve an input payload (True) or output payload (False)

        Returns:
            The retrieved payload, or None if not found
        """
        dataset_name = MODELMESH_INPUT_NAME if is_input else MODELMESH_OUTPUT_NAME

        try:
            async with self.get_lock(dataset_name):
                with H5PYContext(self, dataset_name, "r") as db:
                    if dataset_name not in db:
                        return None

                    dataset = db[dataset_name]
                    if request_id not in dataset.attrs:
                        return None

                    serialized_data = dataset.attrs[request_id]

                    try:
                        payload_dict = pkl.loads(serialized_data)
                        return PartialPayload(**payload_dict)
                    except Exception as e:
                        logger.error(f"Error unpickling payload: {str(e)}")
                        return None
        except MissingH5PYDataException:
            return None
        except Exception as e:
            logger.error(f"Error retrieving ModelMesh payload: {str(e)}")
            return None

    async def get_partial_payload(self, payload_id: str, is_input: bool):
        """Looks up a partial payload by id. Returns None if no matching id exists"""

        # lock to prevent simultaneous read/writes
        partial_dataset_name = PARTIAL_INPUT_NAME if is_input else PARTIAL_OUTPUT_NAME
        async with self.get_lock(partial_dataset_name):
            try:
                with H5PYContext(self, partial_dataset_name, "r") as db:
                    if partial_dataset_name not in db:
                        return None
                    recovered_bytes = db[partial_dataset_name].attrs.get(payload_id)
                    return None if recovered_bytes is None else pkl.loads(recovered_bytes)
            except MissingH5PYDataException:
                return None

    async def delete_modelmesh_payload(self, request_id: str, is_input: bool):
        """
        Delete a stored ModelMesh payload.

        Args:
            request_id: The unique identifier for the inference request
            is_input: Whether to delete an input payload (True) or output payload (False)
        """
        dataset_name = MODELMESH_INPUT_NAME if is_input else MODELMESH_OUTPUT_NAME

        try:
            async with self.get_lock(dataset_name):
                with H5PYContext(self, dataset_name, "a") as db:
                    if dataset_name not in db:
                        return

                    dataset = db[dataset_name]
                    request_ids = list(dataset.attrs["request_ids"])

                    if request_id not in request_ids:
                        return

                    if request_id in dataset.attrs:
                        del dataset.attrs[request_id]

                    request_ids.remove(request_id)
                    dataset.attrs["request_ids"] = request_ids

                    if not request_ids:
                        del db[dataset_name]

            logger.debug(f"Deleted ModelMesh {'input' if is_input else 'output'} payload for request ID: {request_id}")
        except MissingH5PYDataException:
            return
        except Exception as e:
            logger.error(f"Error deleting ModelMesh payload: {str(e)}")
