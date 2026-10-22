"""Canonical storage migration framework.

The public surface deliberately consists of source adapters, one database
target, and a runner. Storage dialect details remain behind those adapters.
"""

from .contracts import (
    DatasetDescriptor,
    PartialPayloadRecord,
    SourceManifest,
)
from .errors import (
    MigrationError,
    MigrationPreflightError,
    MigrationValidationError,
    MigrationWriteError,
)
from .mariadb import MariaDBSource
from .pvc import PVCSource
from .pvc_csv import PVCCSVSource
from .runner import (
    DatasetMigrationResult,
    MigrationConfig,
    MigrationResult,
    MigrationRunner,
    MigrationStatus,
    VerificationResult,
)
from .target import DatabaseTarget

__all__ = [
    "DatabaseTarget",
    "DatasetDescriptor",
    "DatasetMigrationResult",
    "MariaDBSource",
    "MigrationConfig",
    "MigrationError",
    "MigrationPreflightError",
    "MigrationResult",
    "MigrationRunner",
    "MigrationStatus",
    "MigrationValidationError",
    "MigrationWriteError",
    "PVCCSVSource",
    "PVCSource",
    "PartialPayloadRecord",
    "SourceManifest",
    "VerificationResult",
]
