"""PVC-to-MariaDB migration module for TrustyAI Service.

This module handles migrating inference data from PVC (HDF5) storage to MariaDB storage.
Migration is triggered by the DATABASE_ATTEMPT_MIGRATION environment variable and runs
during MariaDBStorage initialization.

Key Features:
- Idempotent migration with progress tracking
- Per-file migration status to support resumption after pod restarts
- Comprehensive error handling and logging
- Observable migration status via database table
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from prometheus_client import Counter

if TYPE_CHECKING:
    from trustyai_service.service.data.storage.maria.maria import MariaDBStorage

from trustyai_service.service.serialization import deserialize_rows

logger = logging.getLogger(__name__)


# Migration status constants
MIGRATION_TYPE_PVC_TO_DB = "PVC_TO_DB"
MIGRATION_STATUS_IN_PROGRESS = "IN_PROGRESS"
MIGRATION_STATUS_COMPLETE = "COMPLETE"
MIGRATION_STATUS_PARTIAL = "PARTIAL"
MIGRATION_STATUS_FAILED = "FAILED"

# Migration configuration defaults
DEFAULT_BATCH_SIZE = 10000  # Rows per batch to prevent OOM on large datasets
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes per batch write operation
DEFAULT_PVC_FOLDER = "/inputs"  # Default PVC mount path

# Log message indicating migration completion (used by test frameworks and operators)
MIGRATION_COMPLETE_MESSAGE = "Migration complete, the PVC is now safe to remove."

# Prometheus metrics for migration observability
migration_files_total = Counter(
    "trustyai_migration_files_total",
    "Total number of HDF5 files discovered for migration",
)
migration_files_success = Counter(
    "trustyai_migration_files_success",
    "Number of HDF5 files successfully migrated",
)
migration_files_failed = Counter(
    "trustyai_migration_files_failed",
    "Number of HDF5 files that failed to migrate",
)
migration_rows_total = Counter(
    "trustyai_migration_rows_total",
    "Total number of data rows migrated from PVC to MariaDB",
)


class PVCToDBMigrator:
    """Handles migration of data from PVC (HDF5) storage to MariaDB.

    This class encapsulates all logic for:
    - Discovering HDF5 files in the PVC mount
    - Reading and deserializing HDF5 datasets
    - Tracking migration progress per file
    - Transferring data to MariaDB storage
    - Logging migration completion for operator awareness

    Design Principles:
    - Idempotent: Safe to run multiple times, skips already-migrated files
    - Resumable: Can resume from checkpoint after pod restart
    - Observable: Migration status persists in database table
    - Fail-safe: Original PVC data remains intact (read-only operations)
    """

    # SQL schema for migration tracking table
    MIGRATION_STATUS_TABLE_SCHEMA = """
        CREATE TABLE IF NOT EXISTS trustyai_migration_status (
            id INT PRIMARY KEY AUTO_INCREMENT,
            migration_type VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP NULL,
            files_processed INT DEFAULT 0,
            total_files INT DEFAULT 0,
            error_message TEXT NULL,
            INDEX idx_migration_type (migration_type),
            INDEX idx_status (status),
            INDEX idx_migration_type_started (migration_type, started_at DESC)
        )
    """

    # SQL schema for per-file migration tracking
    FILE_MIGRATION_STATUS_TABLE_SCHEMA = """
        CREATE TABLE IF NOT EXISTS trustyai_file_migration_status (
            id INT PRIMARY KEY AUTO_INCREMENT,
            migration_id INT NOT NULL,
            file_name VARCHAR(512) NOT NULL,
            dataset_name VARCHAR(255) NOT NULL,
            rows_migrated INT DEFAULT 0,
            completed_at TIMESTAMP NULL,
            error_message TEXT NULL,
            UNIQUE KEY unique_migration_file (migration_id, file_name),
            FOREIGN KEY (migration_id) REFERENCES trustyai_migration_status(id) ON DELETE CASCADE,
            INDEX idx_completed (completed_at)
        )
    """

    def __init__(
        self,
        maria_storage: MariaDBStorage,
        pvc_folder: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize migrator with MariaDB storage instance and PVC folder path.

        :param maria_storage: MariaDBStorage instance to write migrated data to
        :param pvc_folder: Path to PVC mount containing HDF5 files
                          (defaults to STORAGE_DATA_FOLDER env var or DEFAULT_PVC_FOLDER)
        :param batch_size: Number of rows to process per batch
                          (defaults to MIGRATION_BATCH_SIZE env var or DEFAULT_BATCH_SIZE)
        """
        self.maria_storage = maria_storage
        self.connection_manager = maria_storage.connection_manager
        self.pvc_folder = pvc_folder or os.environ.get(
            "STORAGE_DATA_FOLDER", DEFAULT_PVC_FOLDER
        )
        self.batch_size = batch_size or int(
            os.environ.get("MIGRATION_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))
        )
        self._migration_id: int | None = None

    def _create_migration_tables(self) -> None:
        """Create migration status tracking tables if they don't exist."""
        with self.connection_manager as (_conn, cursor):
            cursor.execute(self.MIGRATION_STATUS_TABLE_SCHEMA)
            cursor.execute(self.FILE_MIGRATION_STATUS_TABLE_SCHEMA)
            logger.debug("Migration tracking tables created or verified")

    def _check_migration_already_completed(self) -> bool:
        """Check if a PVC-to-DB migration has already completed.

        :return: True if migration completed successfully in a previous run
        """
        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                "SELECT id, status FROM trustyai_migration_status "
                "WHERE migration_type=? AND status=? "
                "ORDER BY started_at DESC LIMIT 1",
                (MIGRATION_TYPE_PVC_TO_DB, MIGRATION_STATUS_COMPLETE),
            )
            result = cursor.fetchone()
            return result is not None

    def _start_migration_tracking(self, total_files: int) -> int:
        """Create a new migration tracking record and return its ID.

        :param total_files: Total number of HDF5 files to migrate
        :return: Migration ID for tracking this migration run
        """
        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                "INSERT INTO trustyai_migration_status "
                "(migration_type, status, files_processed, total_files) "
                "VALUES (?, ?, 0, ?)",
                (MIGRATION_TYPE_PVC_TO_DB, MIGRATION_STATUS_IN_PROGRESS, total_files),
            )
            migration_id = cursor.lastrowid
            logger.info(
                "Started PVC-to-DB migration tracking (ID=%d, total_files=%d)",
                migration_id,
                total_files,
            )
            return migration_id

    def _update_migration_progress(self, files_processed: int) -> None:
        """Update migration progress in tracking table.

        :param files_processed: Number of files successfully migrated so far
        """
        if self._migration_id is None:
            return

        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                "UPDATE trustyai_migration_status SET files_processed=? WHERE id=?",
                (files_processed, self._migration_id),
            )
            logger.debug("Migration progress: %d files processed", files_processed)

    def _mark_migration_complete(self) -> None:
        """Mark the migration as successfully completed."""
        if self._migration_id is None:
            return

        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                "UPDATE trustyai_migration_status "
                "SET status=?, completed_at=CURRENT_TIMESTAMP "
                "WHERE id=?",
                (MIGRATION_STATUS_COMPLETE, self._migration_id),
            )
            logger.info("Migration marked as complete (ID=%d)", self._migration_id)

    def _mark_migration_partial(self, files_succeeded: int, files_failed: int) -> None:
        """Mark the migration as partially completed with some failures.

        :param files_succeeded: Number of files successfully migrated
        :param files_failed: Number of files that failed to migrate
        """
        if self._migration_id is None:
            return

        error_msg = f"{files_succeeded} files succeeded, {files_failed} files failed"
        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                "UPDATE trustyai_migration_status "
                "SET status=?, error_message=?, completed_at=CURRENT_TIMESTAMP "
                "WHERE id=?",
                (MIGRATION_STATUS_PARTIAL, error_msg, self._migration_id),
            )
            logger.warning(
                "Migration marked as partial (ID=%d): %s",
                self._migration_id,
                error_msg,
            )

    def _mark_migration_failed(self, error_message: str) -> None:
        """Mark the migration as failed with error details.

        :param error_message: Description of the error that caused failure
        """
        if self._migration_id is None:
            return

        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                "UPDATE trustyai_migration_status "
                "SET status=?, error_message=?, completed_at=CURRENT_TIMESTAMP "
                "WHERE id=?",
                (MIGRATION_STATUS_FAILED, error_message, self._migration_id),
            )
            logger.error(
                "Migration marked as failed (ID=%d): %s",
                self._migration_id,
                error_message,
            )

    def _is_file_already_migrated(self, file_name: str) -> bool:
        """Check if a specific HDF5 file has already been migrated.

        :param file_name: Name of the HDF5 file to check
        :return: True if file was previously migrated successfully
        """
        if self._migration_id is None:
            return False

        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                "SELECT id FROM trustyai_file_migration_status "
                "WHERE migration_id=? AND file_name=? AND completed_at IS NOT NULL",
                (self._migration_id, file_name),
            )
            return cursor.fetchone() is not None

    def _mark_file_migrated(
        self, file_name: str, dataset_name: str, rows_migrated: int
    ) -> None:
        """Mark a file as successfully migrated.

        :param file_name: Name of the migrated HDF5 file
        :param dataset_name: Name of the dataset within the file
        :param rows_migrated: Number of rows migrated from this file
        """
        if self._migration_id is None:
            return

        with self.connection_manager as (_conn, cursor):
            # Use INSERT ... ON DUPLICATE KEY UPDATE for idempotence
            cursor.execute(
                "INSERT INTO trustyai_file_migration_status "
                "(migration_id, file_name, dataset_name, rows_migrated, completed_at) "
                "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON DUPLICATE KEY UPDATE "
                "rows_migrated=VALUES(rows_migrated), completed_at=CURRENT_TIMESTAMP",
                (self._migration_id, file_name, dataset_name, rows_migrated),
            )
            logger.debug(
                "Marked file as migrated: %s (dataset=%s, rows=%d)",
                file_name,
                dataset_name,
                rows_migrated,
            )

    def _mark_file_failed(self, file_name: str, error_message: str) -> None:
        """Mark a file migration as failed.

        :param file_name: Name of the HDF5 file that failed to migrate
        :param error_message: Description of the error that occurred
        """
        if self._migration_id is None:
            return

        with self.connection_manager as (_conn, cursor):
            cursor.execute(
                "INSERT INTO trustyai_file_migration_status "
                "(migration_id, file_name, dataset_name, rows_migrated, error_message) "
                "VALUES (?, ?, '', 0, ?) "
                "ON DUPLICATE KEY UPDATE "
                "error_message=VALUES(error_message)",
                (self._migration_id, file_name, error_message),
            )
            logger.error(
                "Marked file as failed: %s - %s",
                file_name,
                error_message,
            )

    def _discover_hdf5_files(self) -> list[Path]:
        """Discover all HDF5 files in the PVC folder.

        :return: List of Path objects for HDF5 files
        :raises FileNotFoundError: If PVC folder doesn't exist
        """
        pvc_path = Path(self.pvc_folder)

        if not pvc_path.exists():
            error_msg = f"PVC folder not found: {self.pvc_folder}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not pvc_path.is_dir():
            error_msg = f"PVC path is not a directory: {self.pvc_folder}"
            logger.error(error_msg)
            raise NotADirectoryError(error_msg)

        hdf5_files = list(pvc_path.glob("*.hdf5"))
        logger.info("Discovered %d HDF5 files in %s", len(hdf5_files), self.pvc_folder)

        return hdf5_files

    def _validate_hdf5_structure(self, hdf5_file: Path) -> tuple[bool, str]:
        """Validate HDF5 file has required structure before migration.

        Checks that the file:
        - Contains at least one dataset
        - Each dataset has required 'column_names' attribute
        - Warns if datasets are empty (allowed but logged for awareness)

        Empty datasets are permitted as they may represent models with no inference
        data yet. The warning helps operators distinguish between intentional empty
        datasets and potential data issues.

        :param hdf5_file: Path to HDF5 file to validate
        :return: Tuple of (is_valid, error_message). error_message is empty string if valid
        """
        try:
            with h5py.File(hdf5_file, "r") as h5f:
                if len(h5f.keys()) == 0:
                    return False, "File contains no datasets"

                for dataset_name in h5f:
                    dataset = h5f[dataset_name]

                    # Check required attributes exist
                    if "column_names" not in dataset.attrs:
                        return (
                            False,
                            f"Dataset '{dataset_name}' missing 'column_names' attribute",
                        )

                    # Warn if dataset is empty (not a failure, might be intentional)
                    if len(dataset) == 0:
                        logger.warning(
                            "Dataset '%s' is empty in %s - will migrate metadata only",
                            dataset_name,
                            hdf5_file.name,
                        )

            return True, ""  # noqa: TRY300

        except Exception as e:
            msg = f"HDF5 validation error: {e}"
            return False, msg

    async def _validate_migration(self, dataset_name: str, expected_rows: int) -> bool:
        """Validate migrated data row count matches source.

        Queries MariaDB to verify the number of rows in the migrated dataset
        matches the expected count from the HDF5 source.

        Note: Validation occurs AFTER data is written to MariaDB. If validation fails,
        data remains in the database but migration is marked as failed. This allows
        manual inspection and recovery rather than silent data loss.

        :param dataset_name: Name of the dataset in MariaDB
        :param expected_rows: Expected row count from HDF5 source
        :return: True if validation passed, False otherwise
        """
        try:
            with self.connection_manager as (_conn, cursor):
                cursor.execute(
                    "SELECT n_rows FROM trustyai_v2_table_reference WHERE dataset_name=?",
                    (dataset_name,),
                )
                result = cursor.fetchone()

                if not result:
                    logger.error(
                        "Validation FAILED: Dataset '%s' not found in MariaDB table reference",
                        dataset_name,
                    )
                    return False

                actual_rows = result[0]
                if actual_rows != expected_rows:
                    logger.error(
                        "Validation FAILED: Dataset '%s' has %d rows in MariaDB, expected %d from HDF5",
                        dataset_name,
                        actual_rows,
                        expected_rows,
                    )
                    return False

                logger.info(
                    "✓ Validation passed: Dataset '%s' has %d rows in MariaDB",
                    dataset_name,
                    expected_rows,
                )
                return True
        except Exception:
            logger.exception(
                "Validation query failed for dataset '%s' (database schema issue?)",
                dataset_name,
            )
            return False

    def _read_hdf5_dataset(
        self, hdf5_file: Path, dataset_name: str
    ) -> tuple[np.ndarray, list[str], bool]:
        """Read a dataset from an HDF5 file with attributes.

        :param hdf5_file: Path to the HDF5 file
        :param dataset_name: Name of the dataset within the file
        :return: Tuple of (data array, column names, is_bytes flag)
        :raises ValueError: If dataset doesn't exist in file
        """
        with h5py.File(hdf5_file, "r") as h5f:
            if dataset_name not in h5f:
                error_msg = (
                    f"Dataset '{dataset_name}' not found in file {hdf5_file.name}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            dataset = h5f[dataset_name]

            # Read HDF5 attributes
            column_names = list(dataset.attrs.get("column_names", []))
            is_bytes = dataset.attrs.get("is_bytes", False)

            # Read data
            data = dataset[:]

            logger.debug(
                "Read dataset '%s' from %s: shape=%s, is_bytes=%s, columns=%d",
                dataset_name,
                hdf5_file.name,
                data.shape,
                is_bytes,
                len(column_names),
            )

            return data, column_names, is_bytes

    async def _write_data_to_maria(
        self,
        dataset_name: str,
        data: np.ndarray,
        column_names: list[str],
    ) -> None:
        """Write migrated data to MariaDB storage.

        :param dataset_name: Name for the dataset in MariaDB
        :param data: Data array to write
        :param column_names: Column names for the dataset
        """
        # Convert numpy array to list for MariaDB write_data
        data_list = data.tolist() if isinstance(data, np.ndarray) else list(data)

        await self.maria_storage.write_data(
            dataset_name=dataset_name,
            new_rows=data_list,
            column_names=column_names,
        )

        logger.debug(
            "Wrote %d rows to MariaDB dataset '%s'",
            len(data_list),
            dataset_name,
        )

    async def _write_data_to_maria_with_timeout(
        self,
        dataset_name: str,
        data: np.ndarray,
        column_names: list[str],
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Write migrated data to MariaDB storage with timeout protection.

        :param dataset_name: Name for the dataset in MariaDB
        :param data: Data array to write
        :param column_names: Column names for the dataset
        :param timeout_seconds: Timeout in seconds (default DEFAULT_TIMEOUT_SECONDS)
        :raises TimeoutError: If write operation exceeds timeout
        """
        try:
            async with asyncio.timeout(timeout_seconds):
                await self._write_data_to_maria(dataset_name, data, column_names)
        except TimeoutError as e:
            error_msg = (
                f"Database write timeout ({timeout_seconds}s) exceeded for dataset '{dataset_name}' "
                f"({len(data)} rows)"
            )
            logger.exception(error_msg)
            raise TimeoutError(error_msg) from e

    async def _migrate_single_file(self, hdf5_file: Path) -> int:
        """Migrate all datasets from a single HDF5 file to MariaDB in batches.

        Validates file structure before migration and validates row counts after.

        :param hdf5_file: Path to the HDF5 file to migrate
        :return: Total number of rows migrated from this file
        :raises ValueError: If validation fails (pre or post migration)
        :raises Exception: Any error during migration (will be caught by caller)
        """
        logger.info("Migrating file: %s", hdf5_file.name)

        # PRE-MIGRATION VALIDATION: Check file structure
        is_valid, error_msg = self._validate_hdf5_structure(hdf5_file)
        if not is_valid:
            msg = f"HDF5 structure validation failed for {hdf5_file.name}: {error_msg}"
            raise ValueError(msg)

        total_rows = 0

        with h5py.File(hdf5_file, "r") as h5f:
            for dataset_name in h5f:
                dataset = h5f[dataset_name]
                dataset_rows = len(dataset)
                column_names = list(dataset.attrs.get("column_names", []))
                is_bytes = dataset.attrs.get("is_bytes", False)

                logger.info(
                    "  Migrating dataset '%s': %d rows (batch size: %d)",
                    dataset_name,
                    dataset_rows,
                    self.batch_size,
                )

                # Migrate in batches to avoid OOM on large datasets
                start_idx = 0
                while start_idx < dataset_rows:
                    end_idx = min(start_idx + self.batch_size, dataset_rows)

                    # Log progress for large datasets
                    if dataset_rows > self.batch_size:
                        logger.info(
                            "    Processing rows %d-%d of %d",
                            start_idx,
                            end_idx,
                            dataset_rows,
                        )

                    # Read batch from HDF5
                    batch_data = dataset[start_idx:end_idx]

                    # Deserialize if data is in serialized format (JSON+gzip)
                    if is_bytes and batch_data.dtype.type in {np.bytes_, np.void}:
                        logger.debug(
                            "Deserializing byte data for dataset '%s' batch %d-%d",
                            dataset_name,
                            start_idx,
                            end_idx,
                        )
                        try:
                            batch_data = deserialize_rows(batch_data)
                        except Exception as deser_error:
                            error_msg = (
                                f"Failed to deserialize dataset '{dataset_name}' "
                                f"batch {start_idx}-{end_idx} in file {hdf5_file.name}: {deser_error}"
                            )
                            logger.exception(error_msg)
                            raise ValueError(error_msg) from deser_error

                    # Write batch to MariaDB with timeout protection
                    await self._write_data_to_maria_with_timeout(
                        dataset_name, batch_data, column_names, timeout_seconds=300
                    )

                    start_idx += self.batch_size

                # POST-MIGRATION VALIDATION: Verify row count matches
                validation_passed = await self._validate_migration(
                    dataset_name, dataset_rows
                )
                if not validation_passed:
                    msg = (
                        f"Post-migration validation failed for dataset '{dataset_name}' "
                        f"in file {hdf5_file.name}"
                    )
                    raise ValueError(msg)

                total_rows += dataset_rows
                logger.info(
                    "  Completed dataset '%s': %d rows migrated",
                    dataset_name,
                    dataset_rows,
                )

        return total_rows

    async def migrate(self) -> None:
        """Execute the full PVC-to-DB migration process.

        This is the main entry point for migration. It handles:
        1. Checking if migration already completed
        2. Discovering HDF5 files
        3. Creating migration tracking tables
        4. Migrating each file with progress tracking
        5. Logging completion message for operator

        Migration is idempotent and resumable - safe to call multiple times.

        :raises FileNotFoundError: If PVC folder doesn't exist
        :raises Exception: Any unexpected error during migration
        """
        # Step 1: Create migration tracking tables
        self._create_migration_tables()

        # Step 2: Check if migration already completed
        if self._check_migration_already_completed():
            logger.info("PVC-to-DB migration already completed in a previous run")
            logger.info(MIGRATION_COMPLETE_MESSAGE)
            return

        # Step 3: Discover HDF5 files
        try:
            hdf5_files = self._discover_hdf5_files()
        except (FileNotFoundError, NotADirectoryError) as e:
            logger.warning("PVC folder not accessible, skipping migration: %s", e)
            # Not an error - PVC might not be mounted during initial DB setup
            return

        if not hdf5_files:
            logger.info(
                "No HDF5 files found in %s, nothing to migrate", self.pvc_folder
            )
            logger.info(MIGRATION_COMPLETE_MESSAGE)
            return

        # Step 4: Start migration tracking
        migration_files_total.inc(len(hdf5_files))
        self._migration_id = self._start_migration_tracking(len(hdf5_files))
        files_processed = 0
        files_failed = 0
        total_rows_migrated = 0

        try:
            # Step 5: Migrate each file
            for hdf5_file in hdf5_files:
                # Skip if already migrated (idempotence)
                if self._is_file_already_migrated(hdf5_file.name):
                    logger.info("Skipping already-migrated file: %s", hdf5_file.name)
                    files_processed += 1
                    continue

                # Migrate the file
                try:
                    rows_migrated = await self._migrate_single_file(hdf5_file)
                    total_rows_migrated += rows_migrated

                    # Mark file as successfully migrated
                    # Extract dataset name from filename (remove _trustyai.hdf5 suffix)
                    dataset_name = hdf5_file.stem.replace("_trustyai", "")
                    self._mark_file_migrated(
                        hdf5_file.name, dataset_name, rows_migrated
                    )

                    files_processed += 1
                    self._update_migration_progress(files_processed)

                    # Update Prometheus metrics
                    migration_files_success.inc()
                    migration_rows_total.inc(rows_migrated)

                except Exception as file_error:
                    logger.exception(
                        "Failed to migrate file %s",
                        hdf5_file.name,
                    )
                    # Mark file as failed
                    self._mark_file_failed(hdf5_file.name, str(file_error))
                    files_failed += 1
                    migration_files_failed.inc()
                    # Continue with next file instead of aborting entire migration
                    # This allows partial recovery if only some files are corrupted
                    continue

            # Step 6: Mark migration complete or partial based on failures
            if files_failed == 0:
                self._mark_migration_complete()
                logger.info(
                    "Successfully migrated %d files (%d total rows) from PVC to MariaDB",
                    files_processed,
                    total_rows_migrated,
                )
                logger.info(MIGRATION_COMPLETE_MESSAGE)
            else:
                self._mark_migration_partial(files_processed, files_failed)
                logger.warning(
                    "Partial migration: %d files succeeded, %d files failed (%d total rows migrated)",
                    files_processed,
                    files_failed,
                    total_rows_migrated,
                )
                logger.warning(
                    "Migration partially complete. Review failed files before removing PVC."
                )

        except Exception as migration_error:
            # Mark migration as failed
            error_msg = f"Migration failed: {migration_error}"
            self._mark_migration_failed(error_msg)
            logger.exception("PVC-to-DB migration failed")
            raise
