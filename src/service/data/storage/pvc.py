import asyncio
from typing import List, Dict

import numpy as np
import os
import h5py
import logging
import pickle as pkl

from src.service.utils import list_utils
from .storage_interface import StorageInterface
from src.service.constants import PROTECTED_DATASET_SUFFIX, PARTIAL_PAYLOAD_DATASET_NAME

logger = logging.getLogger(__name__)
COLUMN_NAMES_ATTRIBUTE = "column_names"
COLUMN_ALIAS_ATTRIBUTE = "column_aliases"
BYTES_ATTRIBUTE = "is_bytes"

PARTIAL_INPUT_NAME = PROTECTED_DATASET_SUFFIX + PARTIAL_PAYLOAD_DATASET_NAME + "_inputs"
PARTIAL_OUTPUT_NAME = PROTECTED_DATASET_SUFFIX + PARTIAL_PAYLOAD_DATASET_NAME + "_outputs"


class H5PYContext:
    """Open the corresponding H5PY file for a dataset and manage its context`"""
    def __init__(self, parent_class, dataset_name, mode):
        self.parent_class = parent_class
        self.mode = mode
        self.dataset_name = dataset_name
        self.filename = parent_class._get_filename(self.dataset_name)

    def __enter__(self):
        if self.mode == 'r' and not os.path.exists(self.filename):
            raise MissingH5PYDataException(self.dataset_name)
        self.db = h5py.File(self.filename, mode=self.mode)
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()


class MissingH5PYDataException(Exception):
    """Raised when a dataset that does not exist is accessed """
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
        self.locks = {fname: asyncio.Lock() for
                      fname in os.listdir(self.data_directory) if self.data_file in fname}
        self.global_lock = asyncio.Lock()

    def _get_filename(self, dataset_name):
        """Get the H5PY filename of a particular dataset"""
        if self.one_file_per_dataset:
            return os.path.join(self.data_directory, dataset_name + "_" + self.data_file)
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

    async def _write_raw_data(self, dataset_name: str, new_rows: np.ndarray, column_names: list[str],
                              is_bytes: bool = False) -> None:
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
                    with H5PYContext(self, allocated_dataset_name, 'a') as db:
                        dataset = db[allocated_dataset_name]

                        if dataset.attrs[BYTES_ATTRIBUTE] != is_bytes:  # data storage paradigm mismatch
                            msg = f"Error when saving inference data for {allocated_dataset_name}: "
                            if dataset.attrs[BYTES_ATTRIBUTE]:
                                msg += ("Dataset was previously saved as serialized tabular data, but has "
                                        "now received a purely numeric payload.")
                            else:
                                msg += ("Dataset was previously saved as numeric data, but has now received "
                                        "a serialized tabular payload.")
                            logger.error(msg)
                            raise ValueError(msg)

                        # add new lines to dataset and write new data
                        dataset.resize(existing_shape[0] + inbound_shape[0], axis=0)
                        dataset[existing_shape[0]:] = new_rows
            else:
                existing_shape_str = ", ".join([":"] + [str(x) for x in existing_shape[1:]])
                inbound_shape_str = ", ".join([":"] + [str(x) for x in inbound_shape[1:]])

                raise ValueError(f"Error when saving inference data for {allocated_dataset_name}: "
                                 f"Mismatch between existing data shape=({existing_shape_str}) vs "
                                 f"inbound data shape=({inbound_shape_str})")
        else:  # first observation of inferences from this model
            async with self.get_lock(allocated_dataset_name):
                with H5PYContext(self, allocated_dataset_name, "a") as db:

                    # create new dataset
                    max_shape = [None] + list(new_rows.shape)[1:]  # to-do: tune this value?
                    dataset = db.create_dataset(allocated_dataset_name, data=new_rows, maxshape=max_shape, chunks=True)
                    dataset.attrs[COLUMN_NAMES_ATTRIBUTE] = column_names
                    dataset.attrs[BYTES_ATTRIBUTE] = is_bytes

    async def write_data(self, dataset_name: str, new_rows, column_names: List[str]):
        """Write new data to a dataset, automatically serializing any non-numeric data"""
        if isinstance(new_rows, np.ndarray):
            if not list_utils.contains_non_numeric(new_rows):
                await self._write_raw_data(dataset_name, new_rows, column_names)
            else:
                await self._write_raw_data(dataset_name, list_utils.serialize_rows(new_rows), column_names)
        elif not list_utils.contains_non_numeric(new_rows):
            await self._write_raw_data(dataset_name, np.array(new_rows), column_names)
        else:
            await self._write_raw_data(dataset_name, list_utils.serialize_rows(new_rows), column_names)

    async def _read_raw_data(self, dataset_name: str, start_row: int = None, n_rows: int = None) -> (np.ndarray, List[str]):
        """Read raw data from a dataset- does not deserialize any bytes data"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, "r") as db:
                if allocated_dataset_name in db:
                    start_row = 0 if start_row is None else start_row
                    end_row = None if n_rows is None else start_row + n_rows
                    dataset = db[allocated_dataset_name]
                    if start_row > dataset.shape[0]:
                        logger.warning(f"Requested a data read from start_row={start_row}, but dataset "
                                       f"only has {dataset.shape[0]} rows. An empty array will be returned.")
                    return dataset[start_row:end_row], dataset.attrs[COLUMN_NAMES_ATTRIBUTE]
                else:
                    raise MissingH5PYDataException(allocated_dataset_name)

    async def read_data(self, dataset_name: str, start_row: int = None, n_rows: int = None) -> (np.ndarray, List[str]):
        """Read data from a dataset, automatically deserializing any byte data"""
        read, column_names = (await self._read_raw_data(dataset_name, start_row, n_rows))
        if len(read) and read[0].dtype.type in {np.bytes_, np.void}:
            return list_utils.deserialize_rows(read), column_names
        else:
            return read, column_names

    async def delete_dataset(self, dataset_name: str):
        """Delete dataset data, ignoring non-existent datasets"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            try:
                with H5PYContext(self, dataset_name, 'a') as db:
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
            with H5PYContext(self, dataset_name, 'r') as db:
                if allocated_dataset_name in db:
                    return db[allocated_dataset_name].attrs[COLUMN_NAMES_ATTRIBUTE]
                else:
                    raise MissingH5PYDataException(allocated_dataset_name)

    async def get_aliased_column_names(self, dataset_name: str) -> List[str]:
        """Get an up-to-date set of column names, including any aliases that might have been applied"""
        allocated_dataset_name = self.allocate_valid_dataset_name(dataset_name)
        async with self.get_lock(allocated_dataset_name):
            with H5PYContext(self, dataset_name, 'r') as db:
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
            with H5PYContext(self, dataset_name, 'a') as db:
                curr_names = db[allocated_dataset_name].attrs[COLUMN_NAMES_ATTRIBUTE]
                aliased_names = [name_mapping.get(name, name) for name in curr_names]
                db[allocated_dataset_name].attrs[COLUMN_ALIAS_ATTRIBUTE] = aliased_names

    async def persist_partial_payload(self, payload, is_input: bool):
        """Save a partial payload to disk. Returns None if no matching id exists"""

        # lock to prevent simultaneous read/writes
        partial_dataset_name = PARTIAL_INPUT_NAME if is_input else PARTIAL_OUTPUT_NAME
        async with self.get_lock(partial_dataset_name):
            with H5PYContext(self, partial_dataset_name, 'a',) as db:
                if partial_dataset_name not in db:
                    dataset = db.create_dataset(partial_dataset_name, dtype="f", track_order=True)
                else:
                    dataset = db[partial_dataset_name]
                dataset.attrs[payload.id] = np.void(pkl.dumps(payload))

    async def get_partial_payload(self, payload_id: str, is_input: bool):
        """Looks up a partial payload by id. Returns None if no matching id exists"""

        # lock to prevent simultaneous read/writes
        partial_dataset_name = PARTIAL_INPUT_NAME if is_input else PARTIAL_OUTPUT_NAME
        async with self.get_lock(partial_dataset_name):
            try:
                with H5PYContext(self, partial_dataset_name, "r") as db:
                    if partial_dataset_name not in db:
                        return None
                    else:
                        recovered_bytes = db[partial_dataset_name].attrs.get(payload_id)
                        return None if recovered_bytes is None else pkl.loads(recovered_bytes)
            except MissingH5PYDataException:
                return None


