import os

from src.service.data.storage.maria.legacy_maria_reader import LegacyMariaDBStorageReader
from src.service.data.storage.maria.maria import MariaDBStorage
from src.service.data.storage.pvc import PVCStorage

def get_storage_interface():
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "PVC")
    if storage_format == "PVC":
        return PVCStorage(data_directory=os.environ.get("STORAGE_DATA_FOLDER", "/tmp"), data_file=os.environ.get("STORAGE_DATA_FILENAME", "trustyai.hdf5"))
    elif storage_format == "MARIA":
        return MariaDBStorage(
            user=os.environ.get("DATABASE_USERNAME"),
            password=os.environ.get("DATABASE_PASSWORD"),
            host=os.environ.get("DATABASE_HOST"),
            port=int(os.environ.get("DATABASE_PORT")),
            database=os.environ.get("DATABASE_DATABASE"),
            attempt_migration=True
        )
    else:
        raise ValueError(f"Storage format={storage_format} not yet supported by the Python implementation of the service.")
