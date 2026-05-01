"""Storage backend implementations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.service.data.storage.maria.maria import MariaDBStorage

from src.service.data.storage.pvc import PVCStorage


class GlobalStorageInterface:
    """Singleton holder for global storage interface."""

    _instance: MariaDBStorage | PVCStorage | None = None

    @classmethod
    def get(cls, *, force_reload: bool = False) -> MariaDBStorage | PVCStorage:
        """Get or create the global storage interface singleton.

        :param force_reload: If True, force recreation of the storage interface
        :return: Storage interface instance (PVCStorage or MariaDBStorage)
        """
        if cls._instance is None or force_reload:
            cls._instance = get_storage_interface()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None


def get_global_storage_interface(
    *, force_reload: bool = False
) -> MariaDBStorage | PVCStorage:
    """Get or create the global storage interface singleton.

    :param force_reload: If True, force recreation of the storage interface
    :return: Storage interface instance (PVCStorage or MariaDBStorage)
    """
    return GlobalStorageInterface.get(force_reload=force_reload)


def get_storage_interface() -> MariaDBStorage | PVCStorage:
    """Create a new storage interface based on environment configuration.

    :return: Storage interface instance (PVCStorage or MariaDBStorage)
    :raises ValueError: If storage format is unsupported or dependencies missing
    """
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "PVC")
    if storage_format == "PVC":
        return PVCStorage(
            data_directory=os.environ.get("STORAGE_DATA_FOLDER", "/tmp"),  # noqa: S108  # fallback default for STORAGE_DATA_FOLDER env var
            data_file=os.environ.get("STORAGE_DATA_FILENAME", "trustyai.hdf5"),
        )
    if storage_format == "MARIA":
        try:
            # Import MariaDB storage only when needed (optional dependency)
            from src.service.data.storage.maria.maria import (  # noqa: PLC0415  # lazy import: mariadb is optional
                MariaDBStorage,
            )

            return MariaDBStorage(
                user=os.environ.get("DATABASE_USERNAME"),
                password=os.environ.get("DATABASE_PASSWORD"),
                host=os.environ.get("DATABASE_HOST"),
                port=int(os.environ.get("DATABASE_PORT", "3306")),
                database=os.environ.get("DATABASE_DATABASE"),
                attempt_migration=bool(
                    int(os.environ.get("DATABASE_ATTEMPT_MIGRATION", "0"))
                ),
            )
        except ImportError as e:
            msg = (
                "MariaDB storage requires optional dependencies. "
                "Install with: pip install trustyai-service[mariadb]. "
                f"Error: {e}"
            )
            raise ValueError(msg) from e
    else:
        msg = f"Storage format={storage_format} not yet supported by the Python implementation of the service."
        raise ValueError(msg)
