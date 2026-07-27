"""Storage backend implementations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trustyai_service.service.data.storage.maria.maria import MariaDBStorage

from trustyai_service.service.data.storage.pvc import PVCStorage


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
            data_directory=os.environ.get("STORAGE_DATA_FOLDER", "/tmp"),  # noqa: S108 -- fallback default for STORAGE_DATA_FOLDER env var
            data_file=os.environ.get("STORAGE_DATA_FILENAME", "trustyai.hdf5"),
        )
    if storage_format in ("MARIA", "DATABASE"):
        try:
            # Import MariaDB storage only when needed (optional dependency)
            from trustyai_service.service.data.storage.maria.maria import (  # noqa: PLC0415 -- lazy import: mariadb is optional
                MariaDBStorage,
            )

            # Parse DATABASE_ATTEMPT_MIGRATION with tolerance for boolean strings
            migration_str = os.environ.get("DATABASE_ATTEMPT_MIGRATION", "0").lower()
            attempt_migration = migration_str in ("1", "true", "yes", "on")

            # Support both operator env vars and direct deployment env vars
            # Operator (Quarkus-based): QUARKUS_DATASOURCE_USERNAME/PASSWORD, DATABASE_SERVICE/NAME
            # Direct deployment: DATABASE_USERNAME/PASSWORD, DATABASE_HOST/DATABASE
            user = os.environ.get("DATABASE_USERNAME") or os.environ.get(
                "QUARKUS_DATASOURCE_USERNAME"
            )
            password = os.environ.get("DATABASE_PASSWORD") or os.environ.get(
                "QUARKUS_DATASOURCE_PASSWORD"
            )
            host = os.environ.get("DATABASE_HOST") or os.environ.get("DATABASE_SERVICE")
            database = os.environ.get("DATABASE_DATABASE") or os.environ.get(
                "DATABASE_NAME"
            )

            # Validate required parameters before constructing MariaDBStorage
            missing = []
            if not user:
                missing.append("DATABASE_USERNAME or QUARKUS_DATASOURCE_USERNAME")
            if not password:
                missing.append("DATABASE_PASSWORD or QUARKUS_DATASOURCE_PASSWORD")
            if not host:
                missing.append("DATABASE_HOST or DATABASE_SERVICE")
            if not database:
                missing.append("DATABASE_DATABASE or DATABASE_NAME")
            if missing:
                msg = f"MariaDB storage requires environment variables: {', '.join(missing)}"
                raise ValueError(msg)

            ssl_ca = os.environ.get("DATABASE_TLS_CA_CERT", "/etc/tls/db/ca.crt")

            return MariaDBStorage(
                user=user,
                password=password,
                host=host,
                port=int(os.environ.get("DATABASE_PORT", "3306")),
                database=database,
                ssl_ca=ssl_ca if Path(ssl_ca).exists() else None,
                attempt_migration=attempt_migration,
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
