"""Canonical migration runner with durable progress and verification."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from .errors import MigrationPreflightError, MigrationValidationError, error_detail

if TYPE_CHECKING:
    from .contracts import (
        DatasetDescriptor,
        MigrationSource,
        MigrationTarget,
        PartialPayloadRecord,
        SourceManifest,
        VerificationStorage,
    )

logger = logging.getLogger(__name__)
_MAX_DATASET_NAME_LENGTH = 255


class MigrationStatus(StrEnum):
    """Lifecycle status for one migration run or dataset."""

    DRY_RUN = "DRY_RUN"
    IN_PROGRESS = "IN_PROGRESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class MigrationConfig:
    """Operational controls for a migration run."""

    batch_size: int = 10_000
    require_quiescent_source: bool = True
    # Kept for API compatibility. Incomplete runs replay whole datasets;
    # completed runs are immutable regardless of this flag.
    resume: bool = True
    verify_checksums: bool = True
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Reject unsafe transfer settings early."""
        if self.batch_size <= 0:
            msg = "batch_size must be greater than zero"
            raise ValueError(msg)


@dataclass
class DatasetMigrationResult:
    """Outcome for one logical dataset."""

    name: str
    status: MigrationStatus
    rows_written: int = 0
    error: str | None = None


@dataclass
class MigrationResult:
    """Serializable summary of one migration attempt."""

    status: MigrationStatus
    total_files: int = 0
    total_datasets: int = 0
    total_rows: int = 0
    datasets_written: int = 0
    rows_read: int = 0
    rows_written: int = 0
    max_batch_rows: int = 0
    failed_datasets: list[DatasetMigrationResult] = field(default_factory=list)
    safe_to_remove_source: bool = False
    error: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    """Post-write comparison result."""

    ok: bool
    datasets_compared: int = 0
    rows_compared: int = 0
    partials_compared: int = 0
    source_rows: int = 0
    target_rows: int = 0
    errors: tuple[str, ...] = ()


class MigrationRunner:
    """Move logical datasets from one source adapter to one database target."""

    def __init__(
        self,
        *,
        source: MigrationSource,
        target: MigrationTarget,
        config: MigrationConfig,
    ) -> None:
        """Initialize a runner with explicit source, target, and policy."""
        self.source = source
        self.target = target
        self.config = config
        self._manifest: SourceManifest | None = None
        self._migration_id: str | None = None
        self._previous_run_complete = False

    async def preflight(self) -> SourceManifest:
        """Validate source consistency and target initialization."""
        if self.config.require_quiescent_source and not self.source.quiesced:
            msg = "source must be quiesced or snapshotted before migration"
            raise MigrationPreflightError(msg)
        manifest = await self.source.preflight()
        self._validate_manifest(manifest)
        self._manifest = manifest
        self._migration_id = self._make_migration_id(manifest)
        if self.config.dry_run:
            return manifest
        await self.target.initialize()
        existing_run = await self.target.get_migration_run(self._migration_id)
        self._previous_run_complete = (
            existing_run is not None
            and existing_run.get("status") == MigrationStatus.COMPLETE.value
        )
        if existing_run is None and await self._target_datasets():
            msg = (
                "target database already contains datasets from another migration; "
                "use a new target database or reset it explicitly"
            )
            raise MigrationPreflightError(msg)
        return manifest

    async def run(self) -> MigrationResult:
        """Run or resume a migration and verify it before reporting completion."""
        manifest = await self.preflight()
        logger.info(
            "Migration preflight: %d datasets, %d rows",
            len(manifest.datasets),
            manifest.total_rows,
        )
        result = MigrationResult(
            status=MigrationStatus.IN_PROGRESS,
            total_files=manifest.total_files,
            total_datasets=len(manifest.datasets),
            total_rows=manifest.total_rows,
        )
        if self.config.dry_run:
            await self._validate_without_writes(manifest, result)
            result.status = MigrationStatus.DRY_RUN
            return result

        # A completed cutover no longer owns the destination: the service may
        # have ingested or edited data since then. Never turn a failed audit
        # into permission to clear it, including on a later retry.
        if self._previous_run_complete:
            verification = await self.verify(manifest=manifest)
            if not verification.ok:
                msg = (
                    "completed migration target has diverged; refusing destructive "
                    "replay: " + "; ".join(verification.errors)
                )
                raise MigrationPreflightError(msg)
            result.status = MigrationStatus.COMPLETE
            result.safe_to_remove_source = True
            return result

        if self._migration_id is None:
            msg = "migration id was not initialized during preflight"
            raise MigrationPreflightError(msg)
        if not await self._save_run(manifest, result):
            return result

        for descriptor in manifest.datasets:
            await self._migrate_dataset(descriptor, result)

        await self._migrate_partials(manifest, result)

        try:
            verification = await self.verify(manifest=manifest)
        except Exception as error:  # noqa: BLE001 -- source/connection failure must produce a failed result
            logger.error("Migration verification failed (%s)", type(error).__name__)  # noqa: TRY400 -- tracebacks may expose SQL parameters
            verification = VerificationResult(
                ok=False,
                errors=(f"verification could not finish ({type(error).__name__})",),
            )
        if result.failed_datasets or not verification.ok:
            result.status = (
                MigrationStatus.PARTIAL
                if result.datasets_written or result.rows_written
                else MigrationStatus.FAILED
            )
            if not verification.ok and verification.errors:
                result.error = "; ".join(verification.errors)
            result.safe_to_remove_source = False
        else:
            result.status = MigrationStatus.COMPLETE
            result.safe_to_remove_source = True
        await self._save_run(manifest, result)
        logger.info(
            "Migration finished: %s; %d rows written",
            result.status.value,
            result.rows_written,
        )
        return result

    async def _migrate_partials(
        self, manifest: SourceManifest, result: MigrationResult
    ) -> None:
        """Transfer in-flight payloads while retaining individual failures."""
        for payload in manifest.partial_payloads:
            try:
                await self.target.persist_partial(payload)
            except Exception as error:  # noqa: BLE001 -- isolate one payload failure
                logger.error(  # noqa: TRY400 -- tracebacks may expose SQL parameters
                    "Partial payload transfer failed (%s)", type(error).__name__
                )
                result.failed_datasets.append(
                    DatasetMigrationResult(
                        name=self._partial_name(payload),
                        status=MigrationStatus.FAILED,
                        error=error_detail(error),
                    )
                )

    async def _save_run(
        self, manifest: SourceManifest, result: MigrationResult
    ) -> bool:
        """Fail closed if durable bookkeeping cannot record the outcome."""
        try:
            await self.target.save_migration_run(
                {
                    "migration_id": self._migration_id,
                    "source_type": manifest.source_type,
                    "source_identity": manifest.digest,
                    "status": result.status.value,
                    "total_files": manifest.total_files,
                    "total_datasets": len(manifest.datasets),
                    "total_rows": manifest.total_rows,
                    "error_message": result.error,
                }
            )
        except Exception as error:  # noqa: BLE001 -- a DB outage must not hide the result
            logger.error("Migration run checkpoint failed (%s)", type(error).__name__)  # noqa: TRY400 -- tracebacks may expose SQL parameters
            result.status = MigrationStatus.FAILED
            result.safe_to_remove_source = False
            detail = f"run checkpoint could not be saved ({type(error).__name__})"
            result.error = f"{result.error}; {detail}" if result.error else detail
            return False
        return True

    async def verify(
        self, *, manifest: SourceManifest | None = None
    ) -> VerificationResult:
        """Compare source rows and metadata with the target namespace."""
        expected = manifest or self._manifest
        if expected is None:
            expected = await self.preflight()
        if self.config.verify_checksums:
            current = await self.source.preflight()
            if current.digest != expected.digest:
                return VerificationResult(
                    ok=False,
                    errors=("source changed after preflight",),
                )

        storage = self._target_storage()
        errors: list[str] = []
        datasets_compared = 0
        rows_compared = 0
        partials_compared = 0
        source_rows = 0
        target_rows = 0
        for descriptor in expected.datasets:
            (
                compared,
                compared_rows,
                target_count,
                dataset_errors,
            ) = await self._verify_dataset(descriptor, storage)
            datasets_compared += compared
            rows_compared += compared_rows
            source_rows += descriptor.row_count
            target_rows += target_count
            errors.extend(dataset_errors)
        for payload in expected.partial_payloads:
            partials_compared += 1
            try:
                actual_payload = await storage.get_partial_payload(
                    payload.payload_id,
                    is_input=payload.is_input,
                    is_modelmesh=payload.is_modelmesh,
                )
            except Exception as error:  # noqa: BLE001 -- report per-payload errors
                errors.append(
                    f"partial verification failed for {payload.payload_id}: {error_detail(error)}"
                )
                continue
            if actual_payload is None:
                errors.append(f"missing target partial payload: {payload.payload_id}")
            elif not _values_equal(payload.payload, actual_payload):
                errors.append(f"partial payload mismatch: {payload.payload_id}")

        return VerificationResult(
            ok=not errors and source_rows == target_rows,
            datasets_compared=datasets_compared,
            rows_compared=rows_compared,
            partials_compared=partials_compared,
            source_rows=source_rows,
            target_rows=target_rows,
            errors=tuple(errors),
        )

    async def _verify_dataset(  # noqa: C901, PLR0911 -- each validation stage has a precise outcome
        self, descriptor: DatasetDescriptor, storage: VerificationStorage
    ) -> tuple[int, int, int, list[str]]:
        """Verify one dataset and return aggregate counters and errors."""
        try:
            if not await storage.dataset_exists(descriptor.name):
                return 0, 0, 0, [f"missing target dataset: {descriptor.name}"]
            actual_columns = await storage.get_original_column_names(descriptor.name)
            actual_aliases = await storage.get_aliased_column_names(descriptor.name)
            if actual_columns != list(descriptor.column_names):
                return 0, 0, 0, [f"column mismatch: {descriptor.name}"]
            if actual_aliases != list(descriptor.aliases):
                return 0, 0, 0, [f"alias mismatch: {descriptor.name}"]
            actual_types = await storage.get_column_types(descriptor.name)
            if actual_types != list(descriptor.column_types):
                return 0, 0, 0, [f"type mismatch: {descriptor.name}"]
            if descriptor.service_metadata:
                actual_service_metadata = await storage.get_service_metadata(
                    descriptor.name
                )
                if actual_service_metadata != descriptor.service_metadata:
                    return 0, 0, 0, [f"service metadata mismatch: {descriptor.name}"]
            count = await storage.dataset_rows(descriptor.name)
            if count != descriptor.row_count:
                return (
                    0,
                    0,
                    count,
                    [
                        f"row-count mismatch: {descriptor.name} expected {descriptor.row_count}, got {count}"
                    ],
                )
            rows_compared = 0
            for start in range(0, descriptor.row_count, self.config.batch_size):
                expected_rows = await self.source.read_rows(
                    descriptor,
                    start,
                    min(self.config.batch_size, descriptor.row_count - start),
                )
                actual_rows = await storage.read_data(
                    descriptor.name,
                    start,
                    min(self.config.batch_size, descriptor.row_count - start),
                )
                rows_compared += len(expected_rows)
                if not _rows_equal(expected_rows, actual_rows):
                    return (
                        1,
                        rows_compared,
                        count,
                        [f"row mismatch: {descriptor.name} at {start}"],
                    )
        except Exception as error:  # noqa: BLE001 -- verification reports per-dataset errors
            return (
                0,
                0,
                0,
                [f"verification failed for {descriptor.name}: {error_detail(error)}"],
            )
        else:
            return 1, rows_compared, count, []

    async def _validate_without_writes(
        self, manifest: SourceManifest, result: MigrationResult
    ) -> None:
        """Read every source batch so dry-run catches decode errors."""
        for descriptor in manifest.datasets:
            for start in range(0, descriptor.row_count, self.config.batch_size):
                rows = await self.source.read_rows(
                    descriptor,
                    start,
                    min(self.config.batch_size, descriptor.row_count - start),
                )
                self._validate_batch(descriptor, rows, start)
                result.rows_read += len(rows)
                result.max_batch_rows = max(result.max_batch_rows, len(rows))

    async def _migrate_dataset(
        self, descriptor: DatasetDescriptor, result: MigrationResult
    ) -> None:
        """Transfer one dataset and isolate its failure."""
        if self._migration_id is None:
            msg = "migration id was not initialized during preflight"
            raise MigrationPreflightError(msg)
        rows_written = 0
        target_present = False
        try:
            had_target = await self._target_storage().dataset_exists(descriptor.name)
            await self.target.ensure_dataset(descriptor)
            target_present = True

            # Dataset checkpoints are written only after the complete dataset is
            # durable. Any incomplete or stale target rows must therefore be
            # removed before retrying, otherwise write_data would append them.
            # A freshly-created dataset is already empty. If a retry found an
            # existing dataset, clear its rows so writes start at row 0.
            if had_target:
                await self.target.rollback_dataset(descriptor.name)
                await self.target.ensure_dataset(descriptor)

            for start in range(0, descriptor.row_count, self.config.batch_size):
                rows = await self.source.read_rows(
                    descriptor,
                    start,
                    min(self.config.batch_size, descriptor.row_count - start),
                )
                self._validate_batch(descriptor, rows, start)
                result.rows_read += len(rows)
                result.max_batch_rows = max(result.max_batch_rows, len(rows))
                await self.target.write_batch(descriptor, rows)
                rows_written += len(rows)
                result.rows_written += len(rows)
            await self.target.save_migration_dataset_state(
                {
                    "migration_id": self._migration_id,
                    "dataset_name": descriptor.name,
                    "source_identity": self._manifest.digest if self._manifest else "",
                    "status": MigrationStatus.COMPLETE.value,
                    "rows_written": rows_written,
                    "error_message": None,
                }
            )
            result.datasets_written += 1
            logger.info(
                "Dataset transfer complete: %s; %d rows", descriptor.name, rows_written
            )
        except Exception as error:  # noqa: BLE001 -- isolate one dataset failure
            logger.error(  # noqa: TRY400 -- tracebacks may expose SQL parameters
                "Dataset transfer failed: %s (%s)",
                descriptor.name,
                type(error).__name__,
            )
            if target_present:
                try:
                    await self.target.rollback_dataset(descriptor.name)
                except Exception as rollback_error:  # noqa: BLE001 -- preserve failure context
                    error = RuntimeError(
                        f"{error_detail(error)}; rollback failed: {error_detail(rollback_error)}"
                    )
                else:
                    result.rows_written -= rows_written
            try:
                await self.target.save_migration_dataset_state(
                    {
                        "migration_id": self._migration_id,
                        "dataset_name": descriptor.name,
                        "source_identity": self._manifest.digest
                        if self._manifest
                        else "",
                        "status": MigrationStatus.FAILED.value,
                        "rows_written": 0,
                        "error_message": error_detail(error),
                    }
                )
            except Exception as checkpoint_error:  # noqa: BLE001 -- preserve the original transfer failure
                logger.error(  # noqa: TRY400 -- tracebacks may expose SQL parameters
                    "Dataset failure checkpoint failed: %s (%s)",
                    descriptor.name,
                    type(checkpoint_error).__name__,
                )
                error = RuntimeError(
                    f"{error_detail(error)}; failure checkpoint could not be saved ({type(checkpoint_error).__name__})"
                )
            result.failed_datasets.append(
                DatasetMigrationResult(
                    name=descriptor.name,
                    status=MigrationStatus.FAILED,
                    rows_written=0,
                    error=error_detail(error),
                )
            )

    def _validate_batch(
        self, descriptor: DatasetDescriptor, rows: np.ndarray, start: int
    ) -> None:
        """Require every source batch to satisfy its advertised row range."""
        expected = min(self.config.batch_size, descriptor.row_count - start)
        if len(rows) != expected:
            msg = f"source returned {len(rows)} rows for {descriptor.name} at {start}; expected {expected}"
            raise MigrationValidationError(msg)

    def _target_storage(self) -> VerificationStorage:
        """Get the storage view used for verification and checkpoints."""
        return self.target.storage

    async def _target_datasets(self) -> list[str]:
        """List destination data separately from migration checkpoint tables."""
        return await self.target.list_target_datasets()

    @staticmethod
    def _validate_manifest(manifest: SourceManifest) -> None:
        """Reject malformed source metadata before creating target state."""
        if manifest.total_files < 0:
            msg = "source manifest has a negative file count"
            raise MigrationValidationError(msg)
        if not manifest.datasets and not manifest.partial_payloads:
            msg = (
                "source manifest contains no datasets or partial payloads; "
                "verify the PVC source path and --data-file "
                "(STORAGE_DATA_FILENAME) or the MariaDB source URL and schema"
            )
            raise MigrationValidationError(msg)
        seen_names: set[str] = set()
        for descriptor in manifest.datasets:
            MigrationRunner._validate_descriptor(descriptor, seen_names)
        seen_partials: set[tuple[str, bool]] = set()
        for payload in manifest.partial_payloads:
            MigrationRunner._validate_partial_payload(payload.payload_id)
            key = (payload.payload_id, payload.is_input)
            if key in seen_partials:
                msg = (
                    "source manifest contains duplicate partial payload: "
                    f"{payload.payload_id}"
                )
                raise MigrationValidationError(msg)
            seen_partials.add(key)

    @staticmethod
    def _validate_descriptor(
        descriptor: DatasetDescriptor, seen_names: set[str]
    ) -> None:
        """Validate one source dataset descriptor."""
        if not descriptor.name or len(descriptor.name) > _MAX_DATASET_NAME_LENGTH:
            msg = "source dataset names must contain 1 to 255 characters"
            raise MigrationValidationError(msg)
        if descriptor.name in seen_names:
            msg = f"source manifest contains duplicate dataset: {descriptor.name}"
            raise MigrationValidationError(msg)
        seen_names.add(descriptor.name)
        if not descriptor.column_names:
            msg = f"source dataset '{descriptor.name}' has no columns"
            raise MigrationValidationError(msg)
        if len(descriptor.aliases) != len(descriptor.column_names):
            msg = f"source dataset '{descriptor.name}' has mismatched aliases"
            raise MigrationValidationError(msg)
        if len(descriptor.column_types) != len(descriptor.column_names):
            msg = f"source dataset '{descriptor.name}' has mismatched types"
            raise MigrationValidationError(msg)
        shape = descriptor.shape or (len(descriptor.column_names),)
        if shape[0] != len(descriptor.column_names) or any(
            value <= 0 for value in shape
        ):
            msg = f"source dataset '{descriptor.name}' has an incompatible shape"
            raise MigrationValidationError(msg)
        if descriptor.row_count < 0:
            msg = f"source dataset '{descriptor.name}' has a negative row count"
            raise MigrationValidationError(msg)

    @staticmethod
    def _validate_partial_payload(payload_id: str) -> None:
        """Validate one source partial-payload identifier."""
        if not payload_id or len(payload_id) > _MAX_DATASET_NAME_LENGTH:
            msg = "source partial-payload IDs must contain 1 to 255 characters"
            raise MigrationValidationError(msg)

    @staticmethod
    def _make_migration_id(manifest: SourceManifest) -> str:
        """Derive a stable checkpoint key from the source snapshot identity.

        Target identity is provided separately by the migration namespace that
        prefixes the checkpoint tables.
        """
        value = f"{manifest.source_type}:{manifest.digest}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _partial_name(payload: PartialPayloadRecord) -> str:
        """Return a readable checkpoint/result name for a partial payload."""
        direction = "input" if payload.is_input else "output"
        return f"partial:{direction}:{payload.payload_id}"


def _rows_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    """Compare object arrays while preserving bytes and nested structures."""
    # Single-column sources are stored one-dimensional while the target always
    # normalizes rows to columns; compare both in column form.
    if expected.ndim == 1:
        expected = expected.reshape(-1, 1)
    if actual.ndim == 1:
        actual = actual.reshape(-1, 1)
    expected_rows = [row.tolist() for row in expected]
    actual_rows = [row.tolist() for row in actual]
    return _values_equal(expected_rows, actual_rows)


def _values_equal(expected: object, actual: object) -> bool:  # noqa: PLR0911 -- distinct recursive value kinds
    """Recursively compare values that may contain NumPy objects."""
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        if (
            isinstance(expected, np.ndarray)
            and isinstance(actual, np.ndarray)
            and expected.shape != actual.shape
        ):
            return False
        # equal_nan=True does not support mixed object/string arrays. Normalize
        # arrays recursively so nested tensors use the same scalar NaN policy.
        return _values_equal(
            expected.tolist() if isinstance(expected, np.ndarray) else expected,
            actual.tolist() if isinstance(actual, np.ndarray) else actual,
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _values_equal(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(
            _values_equal(left, right)
            for left, right in zip(expected, actual, strict=True)
        )
    if isinstance(expected, (float, np.floating)) and isinstance(
        actual, (float, np.floating)
    ):
        # JSON round-trips NaN faithfully, but NaN != NaN under plain equality.
        return bool(expected == actual) or (math.isnan(expected) and math.isnan(actual))
    try:
        return bool(expected == actual)
    except (TypeError, ValueError):
        return False
