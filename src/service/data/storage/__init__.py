import os
from src.service.data.storage.pvc import PVCStorage


def get_storage_interface():
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "PVC")
    if storage_format == "PVC":
        return PVCStorage(
            data_directory=os.environ.get("STORAGE_DATA_FOLDER", "/tmp"),
            data_file=os.environ.get("STORAGE_DATA_FILENAME", "trustyai.hdf5"),
        )
    else:
        raise ValueError(
            f"Storage format={storage_format} not yet supported by the Python implementation of the service."
        )
