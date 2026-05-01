"""Persistent Volume Claim (PVC) storage backend implementation using HDF5."""

from __future__ import annotations

import asyncio
import logging
import pickle as pkl  # nosec B403 - Used for internal data serialization only
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np

if TYPE_CHECKING:
    from types import TracebackType

    from h5py._hl.files import File

from src.endpoints.consumer import KServeInferenceRequest, KServeInferenceResponse
from src.service.constants import (
    INPUT_SUFFIX,
    METADATA_SUFFIX,
    OUTPUT_SUFFIX,
    PARTIAL_PAYLOAD_DATASET_NAME,
    PROTECTED_DATASET_SUFFIX,
)
from src.service.data.metadata.storage_metadata import (
    StorageMetadata,
    StorageMetadataConfig,
)
from src.service.data.modelmesh_parser import PartialPayload
from src.service.payloads.service.schema import Schema
from src.service.payloads.service.schema_item import SchemaItem
from src.service.payloads.values.data_type import DataType
from src.service.utils import list_utils

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

    def __init__(
        self, parent_class: PVCStorage, dataset_name: str, mode: Literal["r", "w", "a"]
    ) -> None:
        """Initialize HDF5 file context manager.

        :param parent_class: Parent PVCStorage instance
        :param dataset_name: Name of the dataset
        :param mode: File access mode ('r', 'w', 'a')
        """
        self.parent_class = parent_class
        self.mode = mode
        self.dataset_name = dataset_name
        # Access private method from parent class (internal helper)
        self.filename = parent_class._get_filename(self.dataset_name)

    def __enter__(self) -> File:
        """Enter context manager and open HDF5 file."""
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
        """Exit context manager and close HDF5 file."""
        self.db.close()


class MissingH5PYDataError(Exception):
    """Raised when a dataset that does not exist is accessed."""

    def __init__(self, dataset_name: str) -> None:
        """Initialize the exception with dataset name.

        :param dataset_name: Name of the missing dataset
        """
        self.dataset_name = dataset_name

    def __str__(self) -> str:
        """Return string representation of the exception."""
        return f"No inference data for dataset={self.dataset_name} found."


class PVCStorage(StorageInterface):
    """HDF5-based storage backend for persistent volume claims."""

    def __init__(
        self, data_directory: str, data_file: str = "trustyai_inference_data.hdf5"
    ) -> None:
        """Initialize PVC storage backend.

        :param data_directory: Directory path for data storage
        :param data_file: Name of the HDF5 data file
        """
        self.data_path = str(Path(data_directory) / data_file)
        self.data_directory = data_directory
        self.data_file = data_file

        # save all datasets into the same H5PY or one file per dataset?
        # one_file_per_dataset=True minimizes the ramifications of file corruption
        # one_file_per_dataset=True also allows read/write concurrence between different datasets
        self.one_file_per_dataset = True
        self.locks = {
            file_path.name: asyncio.Lock()
            for file_path in Path(self.data_directory).iterdir()
            if self.data_file in file_path.name
        }
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
        """Given an arbitrary model name inbound from the model server, ensure that it does not conflict with any internal TrustyAI dataset names."""
        if dataset_name.startswith(PROTECTED_DATASET_SUFFIX):
            return dataset_name.replace(PROTECTED_DATASET_SUFFIX, "inference_")
        return dataset_name

    async def dataset_exists(self, dataset_name: str) -> bool:
        """Check if a file exists in the data directory for this dataset."""
        async with self.get_lock(dataset_name):
            try:
                with H5PYContext(self, dataset_name, "r") as db:
                    return dataset_name in db
            except MissingH5PYDataError:
                return False

    def _list_all_datasets_sync(self) -> list[str]:
        return [
            file_path.name.replace(f"_{self.data_file}", "")
            for file_path in Path(self.data_directory).iterdir()
            if self.data_file in file_path.name
        ]

    async def list_all_datasets(self) -> list[str]:
        """List all datasets known by the dataset."""
        return await asyncio.to_thread(self._list_all_datasets_sync)

    async def dataset_rows(self, dataset_name: str) -> int:
        """Return the number of data rows in dataset.

        Raises FileNotFoundError if the dataset does not exist.
        """
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    return db[allocated_dataset_name].shape[0]
                raise MissingH5PYDataError(allocated_dataset_name)

    async def dataset_shape(self, dataset_name: str) -> tuple[int]:
        """Shape of the dataset, returns a FileNotFoundError if the dataset does not exist."""
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
        """Write new data to file.

        Axis 0 of the data is the row dimension, and data shape must
        align on all subsequent axes.
        """
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
                            raise ValueError(msg)

                        # add new lines to dataset and write new data
                        dataset.resize(existing_shape[0] + inbound_shape[0], axis=0)

                        # Ensure new_rows dtype matches existing dataset dtype for HDF5 compatibility
                        # This is necessary when using dynamic void types - we may need to upcast
                        #
                        # NOTE ON VOID TYPE UPGRADE PATH:
                        # New datasets are created with V{MAX_VOID_TYPE_LENGTH} (line 228) to allow
                        # future appends with variable void sizes. However, datasets created before
                        # this change may have smaller void types (e.g., V47).
                        #
                        # Upgrade behavior:
                        # - If new_rows.dtype > existing dataset.dtype: new_rows is downcast (may
                        #   lose data if serialized size exceeds existing dtype size). This is a
                        #   limitation of HDF5's fixed-dtype requirement.
                        # - To upgrade an existing dataset with small void type: manually recreate
                        #   the dataset with V{MAX_VOID_TYPE_LENGTH} using migration scripts.
                        #
                        # This preserves backward compatibility while allowing optimal storage for
                        # new datasets.
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
                                    "Recreate or migrate the dataset before appending."
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
                    max_shape = [
                        None,
                        *list(new_rows.shape)[1:],
                    ]  # to-do: tune this value?

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
        self, dataset_name: str, new_rows: np.ndarray, column_names: list[str]
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
            serialized = list_utils.serialize_rows(new_rows, MAX_VOID_TYPE_LENGTH)
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
        """Read raw data from a dataset -- does not deserialize any bytes data."""
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
                        "Requested a data read from start_row=%s, but dataset "
                        "only has %s rows. An empty array will be returned.",
                        start_row,
                        dataset.shape[0],
                    )
                return dataset[start_row:end_row]  # type: ignore[bad-index]

    async def read_data(
        self, dataset_name: str, start_row: int = 0, n_rows: int | None = None
    ) -> np.ndarray:
        """Read data from a dataset, automatically deserializing any byte data."""
        read = await self._read_raw_data(dataset_name, start_row, n_rows)
        if len(read) and read[0].dtype.type in {np.bytes_, np.void}:
            return list_utils.deserialize_rows(read)
        return read

    async def delete_dataset(self, dataset_name: str) -> None:
        """Delete dataset data, ignoring non-existent datasets."""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            # Check if HDF5 file exists before opening to prevent phantom file creation
            # Opening in "a" mode creates the file if it doesn't exist
            filename = Path(self.data_directory) / self.data_file
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
                aliased_names = [name_mapping.get(name, name) for name in curr_names]  # type: ignore[not-iterable]
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

    async def get_metadata(self, model_id: str) -> StorageMetadata | None:
        """Get metadata for a specific model including shapes, column names, etc."""
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
                    "columnNames": list(input_names) if input_names is not None else [],  # type: ignore[bad-argument-type]
                    "aliasedNames": list(aliased_input_names)  # type: ignore[bad-argument-type]
                    if aliased_input_names is not None
                    else [],
                }
            except Exception:  # Broad catch intentional: input metadata errors should not break entire metadata retrieval
                logger.exception("Error getting input metadata for %s", model_id)

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
            except Exception:  # Broad catch intentional: output metadata errors should not break entire metadata retrieval
                logger.exception("Error getting output metadata for %s", model_id)

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
            except Exception:  # Broad catch intentional: metadata info errors should not break entire metadata retrieval
                logger.exception("Error getting metadata info for %s", model_id)

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
                        name=col_name,  # type: ignore[bad-argument-type]
                        column_index=idx,
                    )

                input_schema = Schema(items=schema_items, name_mapping=name_mapping)  # type: ignore[bad-argument-type]

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
                StorageMetadataConfig(
                    model_id=model_id,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    observations=observations,
                    recorded_inferences=observations > 0,
                )
            )

            logger.info(
                "Created StorageMetadata for %s: observations=%s, recorded_inferences=%s",
                model_id,
                observations,
                observations > 0,
            )

        except Exception:  # Broad catch intentional: StorageMetadata creation errors should return None, not crash
            logger.exception("Error creating StorageMetadata for %s", model_id)
            return None
        else:
            return storage_metadata

    async def persist_partial_payload(
        self,
        payload: PartialPayload | KServeInferenceRequest | KServeInferenceResponse,
        payload_id: str,
        *,
        is_input: bool,
    ) -> None:
        """Save a KServe or ModelMesh payload to disk."""
        dataset_name = PARTIAL_INPUT_NAME if is_input else PARTIAL_OUTPUT_NAME
        serialized_data = pkl.dumps(payload.model_dump())
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
            except (
                Exception
            ):  # Broad catch intentional: log context before re-raising storage error
                logger.exception(
                    "Error storing %s payload",
                    "ModelMesh" if is_modelmesh else "KServe",
                )
                raise

    async def get_partial_payload(
        self, payload_id: str, *, is_input: bool, is_modelmesh: bool
    ) -> PartialPayload | KServeInferenceRequest | KServeInferenceResponse | None:
        """Retrieve a partial payload from HDF5 storage.

        SECURITY NOTE: This function deserializes pickled data from HDF5 storage.
        Data must originate from trusted internal sources only (stored via save_partial_payload).
        Do not use with user-supplied or external data.
        """
        dataset_name = PARTIAL_INPUT_NAME if is_input else PARTIAL_OUTPUT_NAME

        try:
            async with self.get_lock(dataset_name):
                with H5PYContext(self, dataset_name, "r") as db:
                    # Check dataset and payload existence
                    if (
                        dataset_name not in db
                        or payload_id not in db[dataset_name].attrs
                    ):
                        return None

                    serialized_data = db[dataset_name].attrs[payload_id]
                    payload_dict = pkl.loads(serialized_data)  # noqa: S301  # type: ignore[arg-type]  # nosec B301
                    if is_modelmesh:
                        return PartialPayload(**payload_dict)
                    if is_input:  # kserve input
                        return KServeInferenceRequest(**payload_dict)
                    # kserve output
                    return KServeInferenceResponse(**payload_dict)
        except (
            MissingH5PYDataError,
            Exception,
        ):  # Broad catch intentional: retrieval errors should return None, not crash
            logger.exception(
                "Error retrieving %s payload",
                "ModelMesh" if is_modelmesh else "KServe",
            )
            return None

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

                    if payload_id in dataset.attrs:
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
        except Exception:  # Broad catch intentional: deletion errors should not crash
            logger.exception("Error deleting payload")

    async def persist_modelmesh_payload(
        self, payload: PartialPayload, request_id: str, *, is_input: bool
    ) -> None:
        """Persist a ModelMesh partial payload to storage."""
        await self.persist_partial_payload(payload, request_id, is_input=is_input)

    async def get_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> PartialPayload | None:
        """Retrieve a ModelMesh partial payload from storage."""
        return await self.get_partial_payload(
            request_id, is_input=is_input, is_modelmesh=True
        )

    async def delete_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> None:
        """Delete a ModelMesh partial payload from storage."""
        await self.delete_partial_payload(request_id, is_input=is_input)
