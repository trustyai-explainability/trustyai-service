"""MariaDB storage backend for TrustyAI inference data.

Thin :class:`SQLStorage` subclass: the shared SQLAlchemy Core base owns all
storage control flow; this adapter builds the MariaDB engine (MariaDB
Connector/Python driver) and keeps the MariaDB-only migration machinery
(PVC-to-DB and legacy v1 schema upgrade).

.. deprecated::
    MariaDB is deprecated in favor of the PostgreSQL backend and will be removed
    once a MariaDB-to-PostgreSQL migration tool ships and a release has passed.
    See ``docs/sql-storage-backends-plan.md``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from trustyai_service.service.data.storage.maria.legacy_maria_reader import (
    LegacyMariaDBStorageReader,
)
from trustyai_service.service.data.storage.maria.pvc_migration import PVCToDBMigrator
from trustyai_service.service.data.storage.maria.utils import MariaConnectionManager
from trustyai_service.service.data.storage.sql.base import SQLStorage
from trustyai_service.service.data.storage.sql.engine import (
    build_engine,
    mariadb_connect_args,
    mariadb_url,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class MariaDBStorage(SQLStorage):
    """MariaDB backend built on the shared SQLAlchemy Core base.

    .. deprecated:: Use :class:`PostgreSQLStorage` instead.
    """

    _backend_name = "MariaDB"

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        database: str,
        *,
        ssl_ca: str | None = None,
        attempt_migration: bool = True,
    ) -> None:
        """Initialize MariaDB storage, create schema tables, and schedule migration."""
        logger.warning(
            "The MariaDB storage backend is DEPRECATED and will be removed in a "
            "future release. Migrate to the PostgreSQL backend "
            "(SERVICE_STORAGE_FORMAT=POSTGRESQL). "
            "See docs/sql-storage-backends-plan.md."
        )
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

        engine = build_engine(
            mariadb_url(user, password, host, port, database),
            connect_args=mariadb_connect_args(ssl_ca),
        )
        super().__init__(engine)

        # Retained for the raw-SQL migration helpers (PVC-to-DB, legacy v1 reader),
        # which operate outside the SQLAlchemy Core layer.
        self.connection_manager = MariaConnectionManager(
            user, password, host, port, database, ssl_ca=ssl_ca
        )

        self._migration_task: asyncio.Task | None = None
        if attempt_migration:
            # Attempt to schedule migration to run asynchronously if event loop is available
            try:
                loop = asyncio.get_running_loop()
                self._migration_task = loop.create_task(self._run_migration())
                self._migration_task.add_done_callback(self._on_migration_done)
            except RuntimeError:
                # No event loop running - run migration synchronously
                asyncio.run(self._run_migration())

    @staticmethod
    def _on_migration_done(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Migration failed.", exc_info=exc)

    # === MIGRATORS ================================================================================
    async def _run_migration(self) -> None:
        """Determine which migration to run: PVC-to-DB or legacy DB schema upgrade.

        Migration priority:
        1. PVC-to-DB migration if HDF5 files exist in PVC folder
        2. Legacy DB schema v1->v2 migration if legacy tables exist
        3. No migration if neither condition is met
        """
        # Check for PVC data first (higher priority)
        pvc_folder = os.environ.get("STORAGE_DATA_FOLDER", "/inputs")
        pvc_path = Path(pvc_folder)

        if pvc_path.exists() and pvc_path.is_dir():
            # Check if any HDF5 files exist
            hdf5_files = list(pvc_path.glob("*.hdf5"))
            if hdf5_files:
                logger.info(
                    "Detected %d HDF5 files in %s, starting PVC-to-DB migration",
                    len(hdf5_files),
                    pvc_folder,
                )
                await self._migrate_from_pvc()
                return

        # No PVC data found, check for legacy DB schema migration
        logger.info("No PVC data found, checking for legacy DB schema migration")
        await self._migrate_from_legacy_db()

    async def _migrate_from_pvc(self) -> None:
        """Execute PVC-to-DB migration using PVCToDBMigrator."""
        migrator = PVCToDBMigrator(
            maria_storage=self,
            pvc_folder=os.environ.get("STORAGE_DATA_FOLDER"),
        )
        await migrator.migrate()

    async def _migrate_from_legacy_db(self) -> None:
        """Execute legacy DB schema v1->v2 migration."""
        legacy_reader = LegacyMariaDBStorageReader(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        if legacy_reader.legacy_data_exists():
            logger.info(
                "Legacy TrustyAI v1 data exists in database, checking if a migration is necessary."
            )
            await legacy_reader.migrate_data(self)
