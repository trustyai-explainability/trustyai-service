"""Storage backend implementations."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.service.data.storage.storage_interface import StorageInterface

from src.service.data.storage.db.db_storage import DBStorage
from src.service.data.storage.pvc import PVCStorage

logger = logging.getLogger(__name__)


class GlobalStorageInterface:
    """Singleton holder for global storage interface."""

    _instance: StorageInterface | None = None

    @classmethod
    def get(cls, *, force_reload: bool = False) -> StorageInterface:
        """Get or create the global storage interface singleton."""
        if cls._instance is None or force_reload:
            cls._instance = get_storage_interface()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None


def get_global_storage_interface(*, force_reload: bool = False) -> StorageInterface:
    """Get or create the global storage interface singleton."""
    return GlobalStorageInterface.get(force_reload=force_reload)


def _create_legacy_mariadb_storage() -> StorageInterface:
    """Create MariaDBStorage using the legacy synchronous mariadb connector."""
    from src.service.data.storage.maria.maria import (  # noqa: PLC0415
        MariaDBStorage,
    )

    migration_str = os.environ.get("DATABASE_ATTEMPT_MIGRATION", "0").lower()
    attempt_migration = migration_str in ("1", "true", "yes", "on")

    user = os.environ.get("DATABASE_USERNAME") or os.environ.get(
        "QUARKUS_DATASOURCE_USERNAME"
    )
    password = os.environ.get("DATABASE_PASSWORD") or os.environ.get(
        "QUARKUS_DATASOURCE_PASSWORD"
    )
    host = os.environ.get("DATABASE_HOST") or os.environ.get("DATABASE_SERVICE")
    database = os.environ.get("DATABASE_DATABASE") or os.environ.get("DATABASE_NAME")

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


def get_storage_interface() -> StorageInterface:
    """Create a new storage interface based on environment configuration.

    Supported formats:
    - PVC: HDF5-based file storage (default)
    - MARIA / DATABASE: MariaDB (legacy sync or async SQLAlchemy)
    - POSTGRES / POSTGRESQL: PostgreSQL via SQLAlchemy async (asyncpg)
    """
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "PVC")

    if storage_format == "PVC":
        return PVCStorage(
            data_directory=os.environ.get("STORAGE_DATA_FOLDER", "/tmp"),  # noqa: S108
            data_file=os.environ.get("STORAGE_DATA_FILENAME", "trustyai.hdf5"),
        )

    if storage_format in ("POSTGRES", "POSTGRESQL"):
        from src.service.data.storage.db.engine import (  # noqa: PLC0415
            create_db_engine,
        )

        engine = create_db_engine(storage_format)
        return DBStorage(engine)

    if storage_format in ("MARIA", "DATABASE"):
        try:
            return _create_legacy_mariadb_storage()
        except ImportError:
            logger.info("Legacy mariadb connector not available, using async DBStorage")
            from src.service.data.storage.db.engine import (  # noqa: PLC0415
                create_db_engine,
            )

            engine = create_db_engine(storage_format)
            return DBStorage(engine)

    msg = f"Unsupported storage format: {storage_format}"
    raise ValueError(msg)
