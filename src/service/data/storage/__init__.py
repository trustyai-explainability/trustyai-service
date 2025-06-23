import asyncio
import os

from src.service.data.storage.pvc import PVCStorage

global_storage_interface = None
_storage_lock = asyncio.Lock()


def get_global_storage_interface(force_reload=False):
    global global_storage_interface

    if global_storage_interface is None or force_reload:
        global_storage_interface = get_storage_interface()
    return global_storage_interface


def get_storage_interface():
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "PVC")
    if storage_format == "PVC":
        return PVCStorage(
            data_directory=os.environ.get("STORAGE_DATA_FOLDER", "/tmp"),
            data_file=os.environ.get("STORAGE_DATA_FILENAME", "trustyai.hdf5"),
        )
    elif storage_format == "MARIA":
        try:
            from src.service.data.storage.maria.maria import MariaDBStorage

            return MariaDBStorage(
                user=os.environ.get("DATABASE_USERNAME"),
                password=os.environ.get("DATABASE_PASSWORD"),
                host=os.environ.get("DATABASE_HOST"),
                port=int(os.environ.get("DATABASE_PORT")),
                database=os.environ.get("DATABASE_DATABASE"),
                attempt_migration=bool(int((os.environ.get("DATABASE_ATTEMPT_MIGRATION", "0")))),
            )
        except ImportError as e:
            raise ValueError(
                "MariaDB storage requires optional dependencies. "
                "Install with: pip install trustyai-service[mariadb]. "
                f"Error: {e}"
            )
    else:
        raise ValueError(
            f"Storage format={storage_format} not yet supported by the Python implementation of the service."
        )
