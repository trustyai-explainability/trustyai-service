"""PVC (Persistent Volume Claim) storage backend using HDF5 files."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from types import TracebackType

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.service.constants import (
    INPUT_SUFFIX,
    METADATA_SUFFIX,
    OUTPUT_SUFFIX,
    PARTIAL_PAYLOAD_DATASET_NAME,
    PROTECTED_DATASET_SUFFIX,
)
from trustyai_service.service.data.exceptions import StorageReadError
from trustyai_service.service.data.metadata.storage_metadata import StorageMetadata
from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.data.storage.exceptions import DeserializationError
from trustyai_service.service.payloads.service.schema import Schema
from trustyai_service.service.payloads.service.schema_item import SchemaItem
from trustyai_service.service.payloads.values.data_type import DataType
from trustyai_service.service.serialization import (
    deserialize_model,
    deserialize_rows,
    serialize_model,
    serialize_rows,
)
from trustyai_service.service.utils import list_utils

from .storage_interface import StorageInterface

logger = logging.getLogger(__name__)
COLUMN_NAMES_ATTRIBUTE = "column_names"
COLUMN_ALIAS_ATTRIBUTE = "column_aliases"
BYTES_ATTRIBUTE = "is_bytes"

PARTIAL_INPUT_NAME = PROTECTED_DATASET_SUFFIX + PARTIAL_PAYLOAD_DATASET_NAME + "_inputs"
PARTIAL_OUTPUT_NAME = (
    PROTECTED_DATASET_SUFFIX + PARTIAL_PAYLOAD_DATASET_NAME + "_outputs"
)
MAX_VOID_TYPE_LENGTH = 1024


class H5PYContext:
    """Open the corresponding H5PY file for a dataset and manage its context."""

    def __init__(self, parent_class: PVCStorage, dataset_name: str, mode: str) -> None:
        """Initialize context with parent storage, dataset name, and file mode."""
        self.parent_class = parent_class
        self.mode = mode
        self.dataset_name = dataset_name
        self.filename = parent_class._get_filename(self.dataset_name)

    def __enter__(self) -> h5py.File:
        """Open the HDF5 file and return the file handle."""
        if self.mode == "r" and not Path(self.filename).exists():
            raise MissingH5PYDataError(self.dataset_name)
        self.db = h5py.File(self.filename, mode=self.mode)
        return self.db

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the HDF5 file handle."""
        self.db.close()


class MissingH5PYDataError(Exception):
    """Raised when a dataset that does not exist is accessed."""

    def __init__(self, dataset_name: str) -> None:
        """Initialize with the missing dataset name."""
        self.dataset_name = dataset_name

    def __str__(self) -> str:
        """Return a description of the missing dataset."""
        return f"No inference data for dataset={self.dataset_name} found."


class PVCStorage(StorageInterface):
    """HDF5-based storage backend using persistent volume claims."""

    def __init__(
        self, data_directory: str, data_file: str = "trustyai_inference_data.hdf5"
    ) -> None:
        """Initialize PVC storage with directory and file configuration."""
        self.data_path = str(Path(data_directory) / data_file)
        self.data_directory = data_directory
        self.data_file = data_file

        # save all datasets into the same H5PY or one file per dataset?
        # one_file_per_dataset=True minimizes the ramifications of file corruption
        # one_file_per_dataset=True also allows read/write concurrence between different datasets
        self.one_file_per_dataset = True
        data_dir = Path(self.data_directory)
        if data_dir.exists():
            self.locks = {
                entry.name: asyncio.Lock()
                for entry in data_dir.iterdir()
                if self.data_file in entry.name
            }
        else:
            self.locks = {}
        self.global_lock = asyncio.Lock()

    def _get_filename(self, dataset_name: str) -> str:
        """Get the H5PY filename of a particular dataset."""
        if self.one_file_per_dataset:
            return str(Path(self.data_directory) / f"{dataset_name}_{self.data_file}")
        return self.data_path

    def get_lock(self, dataset_name: str) -> asyncio.Lock:
        """Get the per-document lock to prevent simultaneous read/writes on an individual H5PY file."""
        filename = self._get_filename(dataset_name)
        if filename not in self.locks:
            self.locks[filename] = asyncio.Lock()
        return self.locks[filename]

    @staticmethod
    def allocate_valid_dataset_name(dataset_name: str) -> str:
        """Allocate a valid dataset name that does not conflict with internal names."""
        if dataset_name.startswith(PROTECTED_DATASET_SUFFIX):
            return dataset_name.replace(PROTECTED_DATASET_SUFFIX, "inference_")
        return dataset_name

    async def dataset_exists(self, dataset_name: str) -> bool:
        """Check whether a file exists in the data directory for this dataset."""
        async with self.get_lock(dataset_name):
            try:
                with H5PYContext(self, dataset_name, "r") as db:
                    return dataset_name in db
            except MissingH5PYDataError:
                return False

    def _list_all_datasets_sync(self) -> list[str]:
        return [
            entry.name.replace(f"_{self.data_file}", "")
            for entry in Path(self.data_directory).iterdir()
            if self.data_file in entry.name
        ]

    async def list_all_datasets(self) -> list[str]:
        """List all datasets known by the dataset."""
        return await asyncio.to_thread(self._list_all_datasets_sync)

    async def dataset_rows(self, dataset_name: str) -> int:
        """Return the number of data rows in a dataset."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    return db[allocated_dataset_name].shape[0]
                raise MissingH5PYDataError(allocated_dataset_name)

    async def dataset_shape(self, dataset_name: str) -> tuple[int]:
        """Return the shape of the dataset."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    return db[allocated_dataset_name].shape
                raise MissingH5PYDataError(allocated_dataset_name)

    async def _write_raw_data(
        self,
        dataset_name: str,
        new_rows: np.ndarray,
        column_names: list[str],
        *,
        is_bytes: bool = False,
    ) -> None:
        """Write new data to file."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        inbound_shape = list(new_rows.shape)

        # use the dataset_shape function to check both existence of dataset + shape retrieval, to reduce file reads
        try:
            existing_shape = list(await self.dataset_shape(allocated_dataset_name))
            dataset_exists = True
        except MissingH5PYDataError:
            existing_shape = None
            dataset_exists = False

        # Validate serialized rows don't exceed maximum void type length
        # Note: serialize_rows() now uses dynamic void types, so this check is mainly
        # for existing data or data from other sources
        if (
            isinstance(new_rows.dtype, np.dtypes.VoidDType)
            and new_rows.dtype.itemsize > MAX_VOID_TYPE_LENGTH
        ):
            msg = (
                f"The datatype of the array to be serialized is {new_rows.dtype} - "
                f"the largest serializable void type is V{MAX_VOID_TYPE_LENGTH}"
            )
            raise ValueError(msg)

        if dataset_exists:  # if we've already got saved inferences for this model
            if existing_shape[1:] == inbound_shape[1:]:  # shapes match
                async with self.get_lock(allocated_dataset_name):
                    with H5PYContext(self, allocated_dataset_name, "a") as db:
                        dataset = db[allocated_dataset_name]

                        if (
                            dataset.attrs[BYTES_ATTRIBUTE] != is_bytes
                        ):  # data storage paradigm mismatch
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

                        # === VOID TYPE COMPATIBILITY HANDLING ===
                        # HDF5 requires all rows in a dataset to have the same dtype. For serialized
                        # data (void types), this creates a compatibility challenge:
                        #
                        # - NEW DATASETS: Created with V{MAX_VOID_TYPE_LENGTH} to accommodate future
                        #   rows of varying serialized sizes (see line 276 below)
                        #
                        # - LEGACY DATASETS: May have smaller void types (e.g., V47) if created before
                        #   this upgrade. When appending to these datasets:
                        #   * If new_rows.dtype.itemsize ≤ existing: downcast new data to fit (safe)
                        #   * If new_rows.dtype.itemsize > existing: REJECT with error (would corrupt data)
                        #
                        # MIGRATION PATH for datasets with small void types:
                        # 1. Read existing data: `data = await storage.read_data(dataset_name)`
                        # 2. Delete old dataset: `await storage.delete_dataset(dataset_name)`
                        # 3. Recreate with new data: `await storage.write_data(dataset_name, data, column_names)`
                        #    (New dataset will automatically use V{MAX_VOID_TYPE_LENGTH})
                        #
                        # This preserves backward compatibility while allowing optimal storage for new datasets.
                        if (
                            new_rows.dtype != dataset.dtype
                            and isinstance(new_rows.dtype, np.dtypes.VoidDType)
                            and isinstance(dataset.dtype.type, type(np.void))
                        ):
                            # Prevent silent downcast that would corrupt data
                            if new_rows.dtype.itemsize > dataset.dtype.itemsize:
                                msg = (
                                    f"Cannot append rows: serialized data ({new_rows.dtype.itemsize} bytes) "
                                    f"exceeds existing dataset capacity ({dataset.dtype.itemsize} bytes). "
                                    "To fix: migrate dataset using read → delete → write pattern (see comment above)."
                                )
                                raise ValueError(msg)
                            # Both are void types with compatible sizes, cast new data to match existing dataset
                            new_rows = new_rows.astype(dataset.dtype)

                        dataset[existing_shape[0] :] = new_rows
            else:
                existing_shape_str = ", ".join(
                    [":"] + [str(x) for x in existing_shape[1:]]
                )
                inbound_shape_str = ", ".join(
                    [":"] + [str(x) for x in inbound_shape[1:]]
                )

                msg_0 = (
                    f"Error when saving inference data for {allocated_dataset_name}: "
                    f"Mismatch between existing data shape=({existing_shape_str}) vs "
                    f"inbound data shape=({inbound_shape_str})"
                )
                raise ValueError(msg_0)
        else:  # first observation of inferences from this model
            async with self.get_lock(allocated_dataset_name):
                with H5PYContext(self, allocated_dataset_name, "a") as db:
                    # create new dataset
                    max_shape = [None, *list(new_rows.shape)[1:]]

                    # For void types, use MAX_VOID_TYPE_LENGTH to ensure future appends
                    # with different sizes can be accommodated
                    dataset_dtype = new_rows.dtype
                    if isinstance(new_rows.dtype, np.dtypes.VoidDType):
                        dataset_dtype = f"V{MAX_VOID_TYPE_LENGTH}"
                        # Cast data to match dataset dtype
                        new_rows = new_rows.astype(dataset_dtype)

                    dataset = db.create_dataset(
                        allocated_dataset_name,
                        data=new_rows,
                        maxshape=max_shape,
                        chunks=True,
                        dtype=dataset_dtype,
                    )
                    dataset.attrs[COLUMN_NAMES_ATTRIBUTE] = column_names
                    dataset.attrs[BYTES_ATTRIBUTE] = is_bytes

    async def write_data(
        self, dataset_name: str, new_rows: np.ndarray | list, column_names: list[str]
    ) -> None:
        """Write new data to a dataset, automatically serializing any non-numeric data."""
        if isinstance(new_rows, np.ndarray) and not list_utils.contains_non_numeric(
            new_rows
        ):
            await self._write_raw_data(dataset_name, new_rows, column_names)
        elif (
            isinstance(new_rows, np.ndarray)
            and list_utils.contains_non_numeric(new_rows)
        ) or (
            not isinstance(new_rows, np.ndarray)
            and list_utils.contains_non_numeric(new_rows)
        ):
            serialized = serialize_rows(new_rows, MAX_VOID_TYPE_LENGTH)
            arr = np.array(serialized)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            await self._write_raw_data(
                dataset_name,
                arr,
                column_names,
                is_bytes=True,
            )
        else:
            await self._write_raw_data(dataset_name, np.array(new_rows), column_names)

    async def _read_raw_data(
        self, dataset_name: str, start_row: int | None = None, n_rows: int | None = None
    ) -> np.ndarray:
        """Read raw data from a dataset without deserializing bytes data."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name not in db:
                    raise MissingH5PYDataError(allocated_dataset_name)
                start_row = 0 if start_row is None else start_row
                end_row = None if n_rows is None else start_row + n_rows
                dataset = db[allocated_dataset_name]
                if start_row > dataset.shape[0]:
                    logger.warning(
                        "Requested a data read from start_row=%d, but dataset "
                        "only has %d rows. An empty array will be returned.",
                        start_row,
                        dataset.shape[0],
                    )
                return dataset[start_row:end_row]

    async def read_data(
        self, dataset_name: str, start_row: int = 0, n_rows: int | None = None
    ) -> np.ndarray:
        """Read data from a dataset, automatically deserializing any byte data."""
        read = await self._read_raw_data(dataset_name, start_row, n_rows)
        if len(read) and read[0].dtype.type in {np.bytes_, np.void}:
            return deserialize_rows(read)
        return read

    async def delete_dataset(self, dataset_name: str) -> None:
        """Delete dataset data, ignoring non-existent datasets."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            # Check if HDF5 file exists before opening to prevent phantom file creation
            # Opening in "a" mode creates the file if it doesn't exist
            filename = Path(self._get_filename(allocated_dataset_name))
            if not filename.exists():
                return
            try:
                with H5PYContext(self, allocated_dataset_name, "a") as db:
                    if allocated_dataset_name in db:
                        del db[allocated_dataset_name]
                    if allocated_dataset_name in self.locks:
                        del self.locks[allocated_dataset_name]
            except MissingH5PYDataError:
                pass

    async def get_original_column_names(self, dataset_name: str) -> list[str]:
        """Get the original column names associated with this model, prior to any name mapping."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    return db[allocated_dataset_name].attrs[COLUMN_NAMES_ATTRIBUTE]
                raise MissingH5PYDataError(allocated_dataset_name)

    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        """Get an up-to-date set of column names, including any aliases that might have been applied."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    if COLUMN_ALIAS_ATTRIBUTE in db[dataset_name].attrs:
                        return db[allocated_dataset_name].attrs[COLUMN_ALIAS_ATTRIBUTE]
                    return db[allocated_dataset_name].attrs[COLUMN_NAMES_ATTRIBUTE]
                raise MissingH5PYDataError(allocated_dataset_name)

    async def apply_name_mapping(
        self, dataset_name: str, name_mapping: dict[str, str]
    ) -> None:
        """Apply a new name mapping to a dataset."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "a") as db:
                curr_names = db[allocated_dataset_name].attrs[COLUMN_NAMES_ATTRIBUTE]
                aliased_names = [name_mapping.get(name, name) for name in curr_names]
                db[allocated_dataset_name].attrs[COLUMN_ALIAS_ATTRIBUTE] = aliased_names

    async def clear_name_mapping(self, dataset_name: str) -> None:
        """Clear/remove the name mapping for a dataset."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "a") as db:
                if allocated_dataset_name in db:
                    if COLUMN_ALIAS_ATTRIBUTE in db[allocated_dataset_name].attrs:
                        del db[allocated_dataset_name].attrs[COLUMN_ALIAS_ATTRIBUTE]
                        logger.info(
                            "Successfully cleared name mapping for dataset '%s'",
                            allocated_dataset_name,
                        )
                    else:
                        logger.warning(
                            "Attempted to clear name mapping for dataset '%s', "
                            "but '%s' attribute was not found.",
                            allocated_dataset_name,
                            COLUMN_ALIAS_ATTRIBUTE,
                        )
                else:
                    logger.warning(
                        "Attempted to clear name mapping for dataset '%s', "
                        "but dataset was not found in the database.",
                        allocated_dataset_name,
                    )

    async def get_known_models(self) -> list[str]:
        """Get a list of all model IDs that have inference data stored."""
        all_datasets = await self.list_all_datasets()
        logger.info("All datasets found: %s", all_datasets)
        model_ids = set()

        for dataset_name in all_datasets:
            logger.debug("Processing dataset: %s", dataset_name)
            # Skip internal datasets
            if dataset_name.startswith(PROTECTED_DATASET_SUFFIX):
                logger.debug("Skipping internal dataset: %s", dataset_name)
                continue

            # Extract model ID by removing suffixes
            if dataset_name.endswith(INPUT_SUFFIX):
                model_id = dataset_name[: -len(INPUT_SUFFIX)]
                model_ids.add(model_id)
                logger.debug("Found input dataset for model: %s", model_id)
            elif dataset_name.endswith(OUTPUT_SUFFIX):
                model_id = dataset_name[: -len(OUTPUT_SUFFIX)]
                model_ids.add(model_id)
                logger.debug("Found output dataset for model: %s", model_id)
            elif dataset_name.endswith(METADATA_SUFFIX):
                model_id = dataset_name[: -len(METADATA_SUFFIX)]
                model_ids.add(model_id)
                logger.debug("Found metadata dataset for model: %s", model_id)
            else:
                logger.debug(
                    "Dataset doesn't match expected suffixes: %s", dataset_name
                )

        logger.info("Extracted model IDs: %s", list(model_ids))
        return list(model_ids)

    async def get_metadata(self, model_id: str) -> StorageMetadata:
        """Get metadata for a specific model including shapes, column names, etc.

        Returns:
            StorageMetadata object with schemas and observation counts

        Raises:
            StorageReadError: If metadata cannot be retrieved or is corrupted

        Note:
            This matches the Java API contract (throws StorageReadException).
            TODO: MariaDB backend returns dict instead of StorageMetadata (Issue #153)

        """
        input_dataset = model_id + INPUT_SUFFIX
        output_dataset = model_id + OUTPUT_SUFFIX
        metadata_dataset = model_id + METADATA_SUFFIX

        logger.info("Getting metadata for model_id: %s", model_id)
        logger.info(
            "Looking for datasets: input=%s, output=%s, metadata=%s",
            input_dataset,
            output_dataset,
            metadata_dataset,
        )

        # Check which datasets exist
        input_exists = await self.dataset_exists(input_dataset)
        output_exists = await self.dataset_exists(output_dataset)
        metadata_exists = await self.dataset_exists(metadata_dataset)

        logger.info(
            "Dataset existence: input=%s, output=%s, metadata=%s",
            input_exists,
            output_exists,
            metadata_exists,
        )

        metadata = {
            "modelId": model_id,
            "inputData": None,
            "outputData": None,
            "metadataData": None,
        }

        # Get input data metadata
        if input_exists:
            try:
                logger.info("Retrieving input metadata for %s", input_dataset)
                input_shape = await self.dataset_shape(input_dataset)
                input_names = await self.get_original_column_names(input_dataset)
                aliased_input_names = await self.get_aliased_column_names(input_dataset)
                logger.info(
                    "Input metadata: shape=%s, names=%s, aliases=%s",
                    input_shape,
                    input_names,
                    aliased_input_names,
                )
                metadata["inputData"] = {
                    "shape": list(input_shape) if input_shape is not None else [],
                    "columnNames": list(input_names) if input_names is not None else [],
                    "aliasedNames": list(aliased_input_names)
                    if aliased_input_names is not None
                    else [],
                }
            except Exception as e:
                logger.exception("Error getting input metadata for %s", model_id)
                msg = f"Failed to retrieve input metadata for model {model_id}"
                raise StorageReadError(msg) from e

        # Get output data metadata
        if output_exists:
            try:
                logger.info("Retrieving output metadata for %s", output_dataset)
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
                logger.exception("Error getting output metadata for %s", model_id)
                msg = f"Failed to retrieve output metadata for model {model_id}"
                raise StorageReadError(msg) from e

        # Get metadata data info
        if metadata_exists:
            try:
                logger.info("Retrieving metadata info for %s", metadata_dataset)
                metadata_shape = await self.dataset_shape(metadata_dataset)
                metadata_names = await self.get_original_column_names(metadata_dataset)
                metadata["metadataData"] = {
                    "shape": list(metadata_shape) if metadata_shape is not None else [],
                    "columnNames": list(metadata_names)
                    if metadata_names is not None
                    else [],
                }
            except Exception as e:
                logger.exception("Error getting metadata info for %s", model_id)
                msg = f"Failed to retrieve metadata info for model {model_id}"
                raise StorageReadError(msg) from e

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
                    name_mapping = {
                        orig: alias
                        for orig, alias in zip(
                            column_names, aliased_names, strict=False
                        )
                        if orig != alias
                    }

                # Create SchemaItem objects for each column
                schema_items = {}
                for idx, col_name in enumerate(column_names):
                    schema_items[col_name] = SchemaItem(
                        type=DataType.UNKNOWN,  # We don't have type info in PVC storage
                        name=col_name,
                        column_index=idx,
                    )

                input_schema = Schema(items=schema_items, name_mapping=name_mapping)

                # Get observation count from input dataset
                if input_data.get("shape"):
                    observations = (
                        input_data["shape"][0] if len(input_data["shape"]) > 0 else 0
                    )

            # Create output schema if output data exists
            if output_exists and metadata.get("outputData"):
                output_data = metadata["outputData"]
                column_names = output_data.get("columnNames", [])
                aliased_names = output_data.get("aliasedNames", [])

                # Create name mapping from original to aliased names
                name_mapping = {}
                if len(column_names) == len(aliased_names):
                    name_mapping = {
                        orig: alias
                        for orig, alias in zip(
                            column_names, aliased_names, strict=False
                        )
                        if orig != alias
                    }

                # Create SchemaItem objects for each column
                schema_items = {}
                for idx, col_name in enumerate(column_names):
                    schema_items[col_name] = SchemaItem(
                        type=DataType.UNKNOWN,  # We don't have type info in PVC storage
                        name=col_name,
                        column_index=idx,
                    )

                output_schema = Schema(items=schema_items, name_mapping=name_mapping)

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
                recorded_inferences=observations > 0,  # True if we have data
            )

            logger.info(
                "Created StorageMetadata for %s: observations=%d, recorded_inferences=%s",
                model_id,
                observations,
                observations > 0,
            )

        except Exception as e:
            logger.exception("Error creating StorageMetadata for %s", model_id)
            # Raise exception to match Java API contract (throws StorageReadException)
            # Callers (like endpoints) handle this gracefully with try/except
            msg = f"Failed to retrieve metadata for model {model_id}"
            raise StorageReadError(msg) from e
        else:
            return storage_metadata

    async def persist_partial_payload(
        self,
        payload: PartialPayload | KServeInferenceRequest | KServeInferenceResponse,
        payload_id: str,
        *,
        is_input: bool,
    ) -> None:
        """Save a KServe or ModelMesh payload to disk using secure JSON + gzip serialization."""
        dataset_name = PARTIAL_INPUT_NAME if is_input else PARTIAL_OUTPUT_NAME
        serialized_data = serialize_model(payload)
        is_modelmesh = isinstance(payload, PartialPayload)

        async with self.get_lock(dataset_name):
            try:
                with H5PYContext(self, dataset_name, "a") as db:
                    if dataset_name not in db:
                        dataset = db.create_dataset(
                            dataset_name, dtype="f", track_order=True
                        )
                    else:
                        dataset = db[dataset_name]
                    dataset.attrs[payload_id] = np.void(serialized_data)

                logger.debug(
                    "Stored %s %s payload for request ID: %s",
                    "ModelMesh" if is_modelmesh else "KServe",
                    "input" if is_input else "output",
                    payload_id,
                )
            except Exception:
                logger.exception(
                    "Error storing %s payload",
                    "ModelMesh" if is_modelmesh else "KServe",
                )
                raise

    async def get_partial_payload(
        self, payload_id: str, *, is_input: bool, is_modelmesh: bool
    ) -> PartialPayload | KServeInferenceRequest | KServeInferenceResponse | None:
        """Retrieve a partial payload from HDF5 storage.

        Uses JSON + gzip deserialization.
        """
        dataset_name = PARTIAL_INPUT_NAME if is_input else PARTIAL_OUTPUT_NAME

        try:
            async with self.get_lock(dataset_name):
                with H5PYContext(self, dataset_name, "r") as db:
                    if dataset_name not in db:
                        return None

                    dataset = db[dataset_name]
                    if payload_id not in dataset.attrs:
                        return None

                    serialized_data = dataset.attrs[payload_id]

                    try:
                        # Convert to bytes if needed (HDF5 attributes may return numpy arrays)
                        if isinstance(serialized_data, np.ndarray) or not isinstance(
                            serialized_data, bytes
                        ):
                            serialized_data = bytes(serialized_data)

                        # Determine target class based on payload type
                        if is_modelmesh:
                            target_class = PartialPayload
                        elif is_input:  # kserve input
                            target_class = KServeInferenceRequest
                        else:  # kserve output
                            target_class = KServeInferenceResponse

                        return deserialize_model(serialized_data, target_class)
                    except Exception as e:
                        # Deserialization failure indicates data corruption or format issue
                        # This is distinct from "not found" and should be raised to caller
                        logger.exception(
                            "Deserialization failed for payload '%s' (%s, %s)",
                            payload_id,
                            "ModelMesh" if is_modelmesh else "KServe",
                            "input" if is_input else "output",
                        )
                        payload_type = "ModelMesh" if is_modelmesh else "KServe"
                        direction = "input" if is_input else "output"
                        raise DeserializationError(
                            payload_id=payload_id,
                            reason=f"Failed to deserialize {payload_type} "
                            f"{direction} payload from HDF5 storage",
                            original_exception=e,
                        ) from e
        except MissingH5PYDataError:
            # Dataset doesn't exist - this is expected for new payloads
            return None
        except DeserializationError:
            # Re-raise deserialization errors (don't catch our own exception)
            raise
        except Exception:
            # Unexpected storage errors (file system, permissions, etc.)
            logger.exception(
                "Unexpected error retrieving %s payload '%s'",
                "ModelMesh" if is_modelmesh else "KServe",
                payload_id,
            )
            raise

    async def delete_partial_payload(self, payload_id: str, *, is_input: bool) -> None:
        """Delete a stored partial payload.

        Args:
            payload_id: The unique identifier for the inference request
            is_input: Whether to delete an input payload (True) or output payload (False)

        """
        dataset_name = PARTIAL_INPUT_NAME if is_input else PARTIAL_OUTPUT_NAME

        try:
            async with self.get_lock(dataset_name):
                with H5PYContext(self, dataset_name, "a") as db:
                    if dataset_name not in db:
                        return

                    dataset = db[dataset_name]

                    if payload_id not in dataset.attrs:
                        return

                    del dataset.attrs[payload_id]

                    if not dataset.attrs:
                        del db[dataset_name]

            logger.debug(
                "Deleted %s payload for request ID: %s",
                "input" if is_input else "output",
                payload_id,
            )
        except MissingH5PYDataError:
            return
        except Exception:
            logger.exception("Error deleting payload")

    async def persist_modelmesh_payload(
        self, payload: PartialPayload, request_id: str, *, is_input: bool
    ) -> None:
        """Persist a ModelMesh payload to storage."""
        await self.persist_partial_payload(payload, request_id, is_input=is_input)

    async def get_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> PartialPayload | None:
        """Retrieve a ModelMesh payload from storage."""
        return await self.get_partial_payload(
            request_id, is_input=is_input, is_modelmesh=True
        )

    async def delete_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> None:
        """Delete a ModelMesh payload from storage."""
        await self.delete_partial_payload(request_id, is_input=is_input)
