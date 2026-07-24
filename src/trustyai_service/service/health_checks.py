"""Health check implementations for Kubernetes probes.

Provides readiness and liveness checks for OpenShift/Kubernetes deployments.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
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
            now = time.time()
            if key in self.cache:
                cached_value, cached_time = self.cache[key]
                if now - cached_time < self.ttl:
                    self.hits += 1
                    return cached_value

            # Cache miss or expired - compute new value
            self.misses += 1
            value = compute_func()
            self.cache[key] = (value, now)
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
except ValueError:
    logger.warning(
        "Invalid HEALTH_CACHE_TTL value '%s', using default 5 seconds",
        os.getenv("HEALTH_CACHE_TTL"),
    )
    _health_cache_ttl = 5
_health_cache = HealthCache(ttl_seconds=_health_cache_ttl)

# Production mode detection for security features (path redaction)
_is_production = os.getenv("ENVIRONMENT", "").lower() == "production"


class HealthCheck:
    """Individual health check result."""

    def __init__(
        self, name: str, status: str, data: dict[str, Any] | None = None
    ) -> None:
        """Initialize health check result.

        :param name: Name of the health check
        :param status: Status ('ok' or 'error')
        :param data: Optional additional data (e.g., error messages)
        """
        self.name = name
        self.status = status
        self.data = data or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        :return: Dictionary representation of health check
        """
        result: dict[str, Any] = {"name": self.name, "status": self.status}
        if self.data:
            result["data"] = self.data
        return result


class HealthCheckRegistry:
    """Registry for managing health checks."""

    @staticmethod
    def check_storage_readiness() -> HealthCheck:
        """Check if storage backend is accessible.

        For PVC storage: Verifies mount point exists and is writable (cached).
        For MariaDB: Tests database connection (cached).

        Results are cached for 5 seconds to reduce I/O overhead during
        frequent health checks (Kubernetes probes every 10 seconds).

        :return: HealthCheck indicating storage readiness
        """
        try:
            storage_format = os.getenv("SERVICE_STORAGE_FORMAT", "PVC")

            if storage_format == "PVC":
                # Cache PVC checks to reduce disk I/O
                return _health_cache.get_or_compute(
                    "pvc_storage", HealthCheckRegistry._check_pvc_storage
                )
            if storage_format in ("MARIA", "DATABASE"):
                # Cache MariaDB checks to reduce connection overhead
                return _health_cache.get_or_compute(
                    "maria_storage", HealthCheckRegistry._check_maria_storage
                )
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
                {"error": f"Unexpected error: {e!s}"},
            )

    @staticmethod
    def _check_pvc_storage() -> HealthCheck:
        """Check PVC storage accessibility.

        In production, redacts full paths from error messages for security.

        :return: HealthCheck for PVC storage
        """
        storage_path_str = os.getenv("STORAGE_DATA_FOLDER", "/inputs")
        storage_path = Path(storage_path_str)

        if not storage_path.exists():
            # Redact full path in production
            if _is_production:
                error_msg = "Storage path not accessible"
            else:
                error_msg = f"Storage path {storage_path_str} not found"
            return HealthCheck(
                "Storage readiness",
                STATUS_ERROR,
                {"error": error_msg},
            )

        # Verify write access with a test file
        test_file = storage_path / ".health_check"
        try:
            test_file.write_text("health_check")
            test_file.unlink()
            return HealthCheck("Storage readiness", STATUS_OK)
        except (OSError, PermissionError) as e:
            # Redact details in production
            if _is_production:
                error_msg = "Storage not writable"
            else:
                error_msg = f"Storage not writable: {e!s}"
            return HealthCheck(
                "Storage readiness",
                STATUS_ERROR,
                {"error": error_msg},
            )

    @staticmethod
    def _check_maria_storage() -> HealthCheck:
        """Check MariaDB storage accessibility.

        Reuses MariaConnectionManager and MariaDBConfig for consistent
        connection handling (TLS, env var fallbacks, resource cleanup).

        :return: HealthCheck for MariaDB storage
        """
        if not MARIADB_AVAILABLE:
            return HealthCheck(
                "Storage readiness",
                STATUS_ERROR,
                {"error": "MariaDB library not installed (missing 'mariadb' extra)"},
            )

        try:
            from trustyai_service.service.data.storage import MariaDBConfig  # noqa: PLC0415
            from trustyai_service.service.data.storage.maria.utils import (  # noqa: PLC0415
                MariaConnectionManager,
            )

            config = MariaDBConfig()
            mgr = MariaConnectionManager(
                user=config.user,
                password=config.password,
                host=config.host,
                port=config.port,
                database=config.database,
                ssl_ca=config.ssl_ca,
                connect_timeout=2,
            )

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

        except (OSError, TimeoutError) as e:
            logger.warning("Database health check failed: %s", e)
            error_msg = (
                "Database connection failed"
                if _is_production
                else f"Database connection failed: {e!s}"
            )
            return HealthCheck(
                "Storage readiness",
                STATUS_ERROR,
                {"error": error_msg},
            )
        except Exception as e:  # Health check must not crash
            logger.exception("Unexpected error during database health check")
            error_msg = (
                "Unexpected database error"
                if _is_production
                else f"Unexpected database error: {e!s}"
            )
            return HealthCheck(
                "Storage readiness",
                STATUS_ERROR,
                {"error": error_msg},
            )

    @staticmethod
    def check_http_server() -> HealthCheck:
        """Check if HTTP server is running.

        If this endpoint is being called, the server is up.

        :return: HealthCheck indicating HTTP server is up
        """
        return HealthCheck("HTTP server", STATUS_OK)

    @staticmethod
    def check_application_liveness() -> HealthCheck:
        """Check if application is alive.

        Basic liveness check - if we can respond, we're alive.
        More sophisticated checks could be added:
        - Check for deadlocks
        - Verify background threads are running
        - Check memory usage isn't critical

        :return: HealthCheck indicating application is alive
        """
        return HealthCheck("Application", STATUS_OK)


def perform_readiness_checks() -> tuple[str, list[dict[str, Any]]]:
    """Perform all readiness checks.

    Readiness checks verify the service is ready to accept requests:
    - Storage backend is accessible
    - HTTP server is running

    :return: Tuple of (overall_status, list_of_checks)
             overall_status is "ok" if all checks pass, "error" otherwise
    """
    checks = []

    # Storage check
    storage_check = HealthCheckRegistry.check_storage_readiness()
    checks.append(storage_check.to_dict())

    # HTTP server check
    http_check = HealthCheckRegistry.check_http_server()
    checks.append(http_check.to_dict())

    # Determine overall status (DOWN if any check is DOWN)
    overall_status = STATUS_OK
    for check in checks:
        if check["status"] == STATUS_ERROR:
            overall_status = STATUS_ERROR
            break

    return overall_status, checks


def perform_liveness_checks() -> tuple[str, list[dict[str, Any]]]:
    """Perform all liveness checks.

    Liveness checks verify the application is alive and functioning.
    This is lightweight - just confirms we can respond.

    :return: Tuple of (overall_status, list_of_checks)
             overall_status is "ok" if alive, "error" if dead
    """
    checks = []

    # Application liveness check
    app_check = HealthCheckRegistry.check_application_liveness()
    checks.append(app_check.to_dict())

    # Determine overall status
    overall_status = app_check.status

    return overall_status, checks
