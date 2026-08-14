"""Health check implementations for Kubernetes probes.

Provides readiness and liveness checks for OpenShift/Kubernetes deployments.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Status code constants for health checks
STATUS_OK = "ok"
STATUS_ERROR = "error"

# MariaDB is an optional dependency (mariadb extra)
try:
    import mariadb  # type: ignore[import-untyped]  # noqa: F401

    MARIADB_AVAILABLE = True
except ModuleNotFoundError:
    MARIADB_AVAILABLE = False


class HealthCache:
    """TTL-based cache for health check results.

    Reduces overhead by caching health check results for a short duration.
    Kubernetes probes run every 10 seconds, so a 5-second cache still
    detects failures quickly while minimizing I/O operations.

    Tracks cache hits and misses for monitoring purposes.
    """

    def __init__(self, ttl_seconds: int = 5) -> None:
        """Initialize health cache.

        :param ttl_seconds: Time-to-live for cached values in seconds
        """
        self.ttl = ttl_seconds
        self.cache: dict[str, tuple[Any, float]] = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: str, compute_func: Callable[[], Any]) -> Any:  # noqa: ANN401
        """Get cached value or compute and cache a new one.

        Cache is intentionally generic to support any health check return type.

        :param key: Cache key
        :param compute_func: Function to compute value if cache miss
        :return: Cached or computed value
        """
        with self.lock:
            now = time.monotonic()
            if key in self.cache:
                cached_value, cached_time = self.cache[key]
                if now - cached_time < self.ttl:
                    self.hits += 1
                    return cached_value
            self.misses += 1

        value = compute_func()

        with self.lock:
            self.cache[key] = (value, time.monotonic())
        return value

    def stats(self) -> dict[str, int]:
        """Get cache statistics.

        :return: Dictionary with hits and misses counts
        """
        with self.lock:
            return {"hits": self.hits, "misses": self.misses}


# Global health cache instance with configurable TTL (default: 5 seconds)
# Can be overridden via HEALTH_CACHE_TTL environment variable
try:
    _health_cache_ttl = int(os.getenv("HEALTH_CACHE_TTL", "5"))
    if _health_cache_ttl < 0:
        msg = "TTL must be non-negative"
        raise ValueError(msg)
except ValueError:
    logger.warning(
        "Invalid HEALTH_CACHE_TTL value '%s', using default 5 seconds",
        os.getenv("HEALTH_CACHE_TTL"),
    )
    _health_cache_ttl = 5
_health_cache = HealthCache(ttl_seconds=_health_cache_ttl)


def _sanitize_error(generic: str, detail: str) -> str:
    """Return generic message in production, detailed message otherwise."""
    if os.getenv("ENVIRONMENT", "").lower() == "production":
        return generic
    return detail


@dataclass(slots=True)
class HealthCheck:
    """Individual health check result."""

    name: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"name": self.name, "status": self.status}
        if self.data:
            result["data"] = self.data
        return result


def _get_health_connection_manager() -> Any:  # noqa: ANN401
    """Create a MariaConnectionManager configured for health checks."""
    from trustyai_service.service.data.storage import MariaDBConfig  # noqa: PLC0415
    from trustyai_service.service.data.storage.maria.utils import (  # noqa: PLC0415
        MariaConnectionManager,
    )

    config = MariaDBConfig()
    config.validate()
    return MariaConnectionManager(
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        database=config.database,
        ssl_ca=config.ssl_ca,
        connect_timeout=2,
    )


def check_storage_readiness() -> HealthCheck:
    """Check if storage backend is accessible.

    For PVC storage: Verifies mount point exists and is writable (cached).
    For MariaDB: Tests database connection (cached).

    :return: HealthCheck indicating storage readiness
    """
    try:
        storage_format = os.getenv("SERVICE_STORAGE_FORMAT", "PVC")

        if storage_format == "PVC":
            return _health_cache.get_or_compute("pvc_storage", _check_pvc_storage)
        if storage_format in ("MARIA", "DATABASE"):
            return _health_cache.get_or_compute("maria_storage", _check_maria_storage)
        return HealthCheck(
            "Storage readiness",
            STATUS_ERROR,
            {"error": f"Unknown storage format: {storage_format}"},
        )

    except Exception as e:  # Health check must not crash
        logger.exception("Storage readiness check failed")
        return HealthCheck(
            "Storage readiness",
            STATUS_ERROR,
            {
                "error": _sanitize_error(
                    "Unexpected storage error", f"Unexpected error: {e!s}"
                )
            },
        )


def _check_pvc_storage() -> HealthCheck:
    """Check PVC storage accessibility."""
    storage_path_str = os.getenv("STORAGE_DATA_FOLDER", "/tmp")  # noqa: S108 -- fallback default for STORAGE_DATA_FOLDER env var
    storage_path = Path(storage_path_str)

    if not storage_path.exists():
        return HealthCheck(
            "Storage readiness",
            STATUS_ERROR,
            {
                "error": _sanitize_error(
                    "Storage path not accessible",
                    f"Storage path {storage_path_str} not found",
                )
            },
        )

    if not storage_path.is_dir():
        return HealthCheck(
            "Storage readiness",
            STATUS_ERROR,
            {
                "error": _sanitize_error(
                    "Storage path not accessible",
                    f"Storage path {storage_path_str} is not a directory",
                )
            },
        )

    if os.access(storage_path, os.W_OK):
        return HealthCheck("Storage readiness", STATUS_OK)
    return HealthCheck(
        "Storage readiness",
        STATUS_ERROR,
        {
            "error": _sanitize_error(
                "Storage not writable", f"Storage not writable: {storage_path_str}"
            )
        },
    )


def _check_maria_storage() -> HealthCheck:
    """Check MariaDB storage accessibility."""
    if not MARIADB_AVAILABLE:
        return HealthCheck(
            "Storage readiness",
            STATUS_ERROR,
            {"error": "MariaDB library not installed (missing 'mariadb' extra)"},
        )

    import mariadb  # type: ignore[import-untyped]  # noqa: PLC0415

    try:
        mgr = _get_health_connection_manager()

        with mgr as (_conn, cursor):
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

        if result is not None and result[0] == 1:
            return HealthCheck("Storage readiness", STATUS_OK)
        return HealthCheck(
            "Storage readiness",
            STATUS_ERROR,
            {"error": "Database query returned unexpected result"},
        )

    except (mariadb.Error, ValueError) as e:
        logger.warning("Database health check failed: %s", e)
        return HealthCheck(
            "Storage readiness",
            STATUS_ERROR,
            {
                "error": _sanitize_error(
                    "Database connection failed", f"Database connection failed: {e!s}"
                )
            },
        )
    except Exception as e:  # Health check must not crash
        logger.exception("Unexpected error during database health check")
        return HealthCheck(
            "Storage readiness",
            STATUS_ERROR,
            {
                "error": _sanitize_error(
                    "Unexpected database error", f"Unexpected database error: {e!s}"
                )
            },
        )


def check_migration_readiness() -> HealthCheck:  # noqa: PLR0911
    """Check if PVC-to-MariaDB migration is complete.

    Only relevant when SERVICE_STORAGE_FORMAT is MARIA/DATABASE and
    DATABASE_ATTEMPT_MIGRATION is enabled. Returns OK when migration
    is not configured, complete, or partially complete.

    :return: HealthCheck indicating migration readiness
    """
    storage_format = os.getenv("SERVICE_STORAGE_FORMAT", "PVC")
    migration_enabled = os.getenv("DATABASE_ATTEMPT_MIGRATION", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if storage_format not in ("MARIA", "DATABASE") or not migration_enabled:
        return HealthCheck("Migration", STATUS_OK)

    try:
        from trustyai_service.service.data.storage.maria.pvc_migration import (  # noqa: PLC0415
            MIGRATION_STATUS_COMPLETE,
            MIGRATION_STATUS_FAILED,
            MIGRATION_STATUS_IN_PROGRESS,
            MIGRATION_STATUS_PARTIAL,
        )

        mgr = _get_health_connection_manager()

        with mgr as (_conn, cursor):
            cursor.execute(
                "SELECT status FROM trustyai_migration_status "
                "WHERE migration_type IN ('PVC_TO_DB', 'LEGACY_DB') "
                "ORDER BY started_at DESC LIMIT 1"
            )
            result = cursor.fetchone()

        if result is None:
            return HealthCheck(
                "Migration",
                STATUS_ERROR,
                {"error": "Migration not started yet"},
            )

        migration_status = result[0]
        if migration_status == MIGRATION_STATUS_IN_PROGRESS:
            return HealthCheck(
                "Migration",
                STATUS_ERROR,
                {"error": "Data migration in progress"},
            )
        if migration_status == MIGRATION_STATUS_FAILED:
            return HealthCheck(
                "Migration",
                STATUS_ERROR,
                {"error": "Data migration failed"},
            )
        if migration_status == MIGRATION_STATUS_PARTIAL:
            logger.warning("Service ready with partial migration - some files failed")
            return HealthCheck("Migration", STATUS_OK)
        if migration_status == MIGRATION_STATUS_COMPLETE:
            return HealthCheck("Migration", STATUS_OK)
        logger.warning("Unrecognized migration status: %s", migration_status)
        return HealthCheck(
            "Migration",
            STATUS_ERROR,
            {"error": "Unrecognized migration status"},
        )

    except Exception:
        logger.exception("Failed to check migration status")
        return HealthCheck(
            "Migration",
            STATUS_ERROR,
            {"error": "Unable to verify migration status"},
        )


def check_http_server() -> HealthCheck:
    """Check if HTTP server is running.

    If this endpoint is being called, the server is up.

    :return: HealthCheck indicating HTTP server is up
    """
    return HealthCheck("HTTP server", STATUS_OK)


def check_application_liveness() -> HealthCheck:
    """Check if application is alive.

    Basic liveness check - if we can respond, we're alive.

    :return: HealthCheck indicating application is alive
    """
    return HealthCheck("Application", STATUS_OK)


def perform_readiness_checks() -> tuple[str, list[dict[str, Any]]]:
    """Perform all readiness checks.

    :return: Tuple of (overall_status, list_of_checks)
    """
    checks = []

    storage_check = check_storage_readiness()
    checks.append(storage_check.to_dict())

    migration_check = _health_cache.get_or_compute(
        "migration", check_migration_readiness
    )
    checks.append(migration_check.to_dict())

    http_check = check_http_server()
    checks.append(http_check.to_dict())

    overall_status = STATUS_OK
    for check in checks:
        if check["status"] == STATUS_ERROR:
            overall_status = STATUS_ERROR
            break

    return overall_status, checks


def perform_liveness_checks() -> tuple[str, list[dict[str, Any]]]:
    """Perform all liveness checks.

    :return: Tuple of (overall_status, list_of_checks)
    """
    app_check = check_application_liveness()
    overall_status = app_check.status
    return overall_status, [app_check.to_dict()]
