import os
from src.service.data.storage.pvc import PVCStorage

def get_storage_interface():
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT")
    if storage_format == "PVC":
        return PVCStorage(data_directory=os.environ.get("STORAGE_DATA_FOLDER"), data_file=os.environ.get("STORAGE_DATA_FILENAME"))
    else:
        raise ValueError(f"Storage format={storage_format} not yet supported by the Python implementation of the service.")
