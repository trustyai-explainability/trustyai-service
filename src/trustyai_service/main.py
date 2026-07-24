"""TrustyAI service main application entry point and FastAPI configuration."""

import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, Response

if TYPE_CHECKING:
    from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Endpoint routers
from trustyai_service.endpoints.consumer.consumer_endpoint import (
    router as consumer_router,
)
from trustyai_service.endpoints.data.data_upload import router as data_upload_router
from trustyai_service.endpoints.explainers.global_explainer import (
    router as explainers_global_router,
)
from trustyai_service.endpoints.explainers.local_explainer import (
    router as explainers_local_router,
)
from trustyai_service.endpoints.metadata import router as metadata_router
from trustyai_service.endpoints.metrics.batch_mean import router as batch_mean_router
from trustyai_service.endpoints.metrics.drift.compare_means import (
    router as drift_comparemeans_router,
)
from trustyai_service.endpoints.metrics.drift.kolmogorov_smirnov import (
    router as drift_kstest_router,
)
from trustyai_service.endpoints.metrics.fairness.group.dir import router as dir_router
from trustyai_service.endpoints.metrics.fairness.group.spd import router as spd_router
from trustyai_service.endpoints.metrics.metrics_info import (
    router as metrics_info_router,
)

# Middleware
from trustyai_service.middleware.gzip_middleware import GzipRequestMiddleware
from trustyai_service.service.data.storage.maria.pvc_migration import (
    MIGRATION_STATUS_COMPLETE,
    MIGRATION_STATUS_FAILED,
    MIGRATION_STATUS_IN_PROGRESS,
    MIGRATION_STATUS_PARTIAL,
)
from trustyai_service.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)

# Valid storage formats (for environment variable validation)
VALID_STORAGE_FORMATS = {"PVC", "MARIA"}

lm_evaluation_harness_router: "APIRouter | None" = None
try:
    from trustyai_service.endpoints.evaluation.lm_evaluation_harness import router

    lm_evaluation_harness_router = router
except ImportError:
    # LM evaluation harness requires optional 'eval' extra dependencies
    # ImportError (not ModuleNotFoundError) because the module may exist but fail to import
    pass

logging.basicConfig(
    level=logging.INFO,  # Reduce default verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Enable debug logging for TrustyAI components only
logging.getLogger("src").setLevel(logging.DEBUG)
logging.getLogger("__main__").setLevel(logging.DEBUG)

# Migration status cache for readiness probe optimization
# Prevents database query on every health check (typically every 10s)
# Cache TTL: 60 seconds (once migration is complete, it stays complete)
_migration_status_cache: dict[str, Any] = {"status": None, "timestamp": 0}
_MIGRATION_CACHE_TTL = 60.0  # seconds

# Remove noisy HTTP/2 and hypercorn internal logs
logging.getLogger("hpack.hpack").setLevel(logging.WARNING)
logging.getLogger("hypercorn.protocol").setLevel(logging.INFO)
logging.getLogger("hypercorn.access").setLevel(logging.INFO)

# Ensure scheduler debug logging
scheduler_logger = logging.getLogger(
    "trustyai_service.service.prometheus.prometheus_scheduler"
)
scheduler_logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

prometheus_scheduler = get_shared_prometheus_scheduler()


async def schedule_metrics_calculation() -> None:
    """Background task to calculate metrics at regular intervals."""
    while True:
        try:
            await prometheus_scheduler.calculate()
        except (
            Exception
        ):  # Broad catch intentional: scheduler errors should not crash background task
            logger.exception("Error in metrics calculation")

        # Wait for the configured interval
        interval = prometheus_scheduler.service_config.get("metrics_schedule", 30)
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage FastAPI application lifespan with background metrics task.

    :param _app: FastAPI application instance
    :yield: Control during application runtime
    """
    # Start the background metrics calculation task
    task = asyncio.create_task(schedule_metrics_calculation())

    yield

    # Cancel the task on shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Prometheus metrics calculation task cancelled during shutdown")


app = FastAPI(
    title="TrustyAI Service API",
    version="1.0.0rc0",
    description="TrustyAI Service API",
    lifespan=lifespan,
)

# CORS (added first, runs last)
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]  # FastAPI/Starlette middleware typing limitation
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip decompression for KServe agent uploads (added last, runs first)
# This ensures request decompression happens before other middleware
app.add_middleware(GzipRequestMiddleware)


@app.middleware("http")
async def strip_trailing_slash(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Strip trailing slashes to avoid 307 redirects that drop POST bodies."""
    if request.url.path != "/" and request.url.path.endswith("/"):
        request.scope["path"] = request.url.path.rstrip("/")
    return await call_next(request)


# Include all routers
app.include_router(
    consumer_router,
    tags=["{Internal Only} Inference Consumer", "{Internal Only} ModelMesh Consumer"],
)
app.include_router(dir_router, tags=["Fairness Metrics: Group: Disparate Impact Ratio"])
app.include_router(data_upload_router, tags=["Data Upload"])

#   Drift metrics
app.include_router(
    drift_comparemeans_router,
    tags=[
        "Drift Metrics: CompareMeans",
    ],
)
app.include_router(
    drift_kstest_router,
    tags=[
        "Drift Metrics: KSTest",
    ],
)

app.include_router(explainers_global_router, tags=["Explainers: Global"])
app.include_router(explainers_local_router, tags=["Explainers: Local"])
app.include_router(
    spd_router,
    tags=["Fairness Metrics: Group: Statistical Parity Difference"],
)
app.include_router(batch_mean_router, tags=["Metrics: Batch Mean"])
app.include_router(metadata_router, tags=["Service Metadata"])
app.include_router(metrics_info_router, tags=["Metrics Information Endpoint"])

if lm_evaluation_harness_router is not None:
    app.include_router(
        lm_evaluation_harness_router, tags=["LM Evaluation Harness Endpoint"]
    )

# Deprecated endpoints
app.include_router(
    dir_router, prefix="/metrics", tags=["{Legacy}: Disparate Impact Ratio"]
)
app.include_router(
    spd_router,
    prefix="/metrics",
    tags=["{Legacy}: Statistical Parity Difference"],
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning service welcome message.

    :return: Dictionary with welcome message
    """
    return {"message": "Welcome to TrustyAI Explainability Service"}


@app.get("/q/metrics")
async def metrics(_request: Request) -> Response:
    """Prometheus metrics endpoint.

    :param _request: FastAPI request object (unused)
    :return: Prometheus metrics in text format
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _handle_migration_status(status: str) -> JSONResponse | None:
    """Handle migration status and return appropriate readiness response.

    :param status: Migration status (IN_PROGRESS, FAILED, PARTIAL, or COMPLETE)
    :return: JSONResponse if service should be not ready, None if ready
    """
    if status == MIGRATION_STATUS_IN_PROGRESS:
        return JSONResponse(
            content={
                "status": "not_ready",
                "reason": "Data migration in progress",
            },
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
    if status == MIGRATION_STATUS_FAILED:
        return JSONResponse(
            content={
                "status": "not_ready",
                "reason": "Data migration failed",
            },
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
    if status == MIGRATION_STATUS_PARTIAL:
        logger.warning("Service ready with partial migration - some files failed")
        # Fall through to ready (return None)
    # COMPLETE or PARTIAL status - service is ready
    return None


# Readiness probe
@app.get("/q/health/ready")
async def readiness_probe() -> JSONResponse:
    """Kubernetes readiness probe endpoint.

    Blocks pod readiness if DATABASE_ATTEMPT_MIGRATION is enabled and migration
    is still in progress. This ensures the service doesn't receive traffic until
    data migration completes.

    :return: JSON response indicating service is ready (200) or not ready (503)
    """
    # Check if migration is required
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "PVC")

    # Validate storage format
    if storage_format not in VALID_STORAGE_FORMATS:
        logger.warning(
            "Invalid SERVICE_STORAGE_FORMAT '%s', defaulting to PVC. Valid formats: %s",
            storage_format,
            VALID_STORAGE_FORMATS,
        )
        storage_format = "PVC"

    migration_enabled = os.environ.get("DATABASE_ATTEMPT_MIGRATION", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if storage_format == "MARIA" and migration_enabled:
        # Check cache first to avoid DB query on every health check
        current_time = time.time()
        cache_age = current_time - _migration_status_cache["timestamp"]

        # Use cached status if:
        # 1. Cache is fresh (< TTL), OR
        # 2. Migration is COMPLETE (no need to check again)
        if (
            cache_age < _MIGRATION_CACHE_TTL
            or _migration_status_cache["status"] == MIGRATION_STATUS_COMPLETE
        ):
            cached_status = _migration_status_cache["status"]
            if cached_status is not None:
                response = _handle_migration_status(cached_status)
                if response is not None:
                    return response
                # COMPLETE or PARTIAL status - fall through to ready
        else:
            # Cache miss or expired - query database and update cache
            try:
                from trustyai_service.service.data.storage import (  # noqa: PLC0415 -- conditional import based on runtime config
                    get_global_storage_interface,
                )
                from trustyai_service.service.data.storage.maria.maria import (  # noqa: PLC0415 -- conditional import
                    MariaDBStorage,
                )

                storage = get_global_storage_interface()

                # Check if storage is MariaDB (type guard)
                if isinstance(storage, MariaDBStorage):
                    # Query migration status from database
                    with storage.connection_manager as (_conn, cursor):
                        cursor.execute(
                            "SELECT status FROM trustyai_migration_status "
                            "WHERE migration_type IN ('PVC_TO_DB', 'LEGACY_DB') "
                            "ORDER BY started_at DESC LIMIT 1"
                        )
                        result = cursor.fetchone()

                        if result:
                            migration_status = result[0]
                            # Update cache
                            _migration_status_cache["status"] = migration_status
                            _migration_status_cache["timestamp"] = current_time

                            response = _handle_migration_status(migration_status)
                            if response is not None:
                                return response
                            # Migration complete - proceed to ready state
                        else:
                            # No migration row exists yet - migration not started
                            # Treat as not ready to prevent traffic before migration begins
                            return JSONResponse(
                                content={
                                    "status": "not_ready",
                                    "reason": "Migration not started yet",
                                },
                                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                            )

            except Exception as e:
                # If we can't check migration status, assume not ready
                # This prevents traffic during migration issues
                logger.exception("Failed to check migration status")
                return JSONResponse(
                    content={
                        "status": "not_ready",
                        "reason": f"Unable to verify migration status: {e}",
                    },
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )

    # No migration required or migration completed successfully
    return JSONResponse(content={"status": "ready"}, status_code=HTTPStatus.OK)


# Liveness probe endpoint
@app.get("/q/health/live")
async def liveness_probe() -> JSONResponse:
    """Kubernetes liveness probe endpoint.

    :return: JSON response indicating service is alive
    """
    return JSONResponse(content={"status": "live"}, status_code=HTTPStatus.OK)


def get_tls_config() -> dict[str, Any] | None:
    """Get TLS configuration for the service.

    Returns SSL configuration if certificates are available, None
    otherwise.
    """
    cert_file = os.getenv("TLS_CERT_FILE", "/etc/tls/internal/tls.crt")
    key_file = os.getenv("TLS_KEY_FILE", "/etc/tls/internal/tls.key")

    cert_path = Path(cert_file)
    key_path = Path(key_file)

    if cert_path.exists() and key_path.exists():
        logger.info("TLS certificates found at %s and %s", cert_file, key_file)
        return {
            "ssl_keyfile": str(key_path),
            "ssl_certfile": str(cert_path),
            "ssl_version": 2,  # TLS v1.2+
        }
    logger.info("TLS certificates not found, running in HTTP mode")
    return None


async def run_server() -> None:
    """Run hypercorn server with both HTTP and HTTPS binds."""
    # Get TLS configuration
    tls_config = get_tls_config()

    # Configure server settings
    host_https = "0.0.0.0"  # noqa: S104  # intentional: Kubernetes service binding
    host_http = (
        "127.0.0.1"  # Keep loopback-only for security (kube-rbac-proxy forwards here)
    )
    http_port = int(os.getenv("HTTP_PORT", "8080"))
    ssl_port = int(os.getenv("SSL_PORT", "4443"))

    # Create hypercorn config
    config = Config()

    # HTTP for kube-rbac-proxy (plain HTTP on insecure_bind)
    config.insecure_bind = [f"{host_http}:{http_port}"]
    logger.info("Binding HTTP on %s:%s for kube-rbac-proxy", host_http, http_port)

    # Configure for HTTP/1.1 compatibility and proper keep-alive
    config.h11_max_incomplete_size = 16 * 1024 * 1024  # 16MB for large requests
    config.keep_alive_timeout = float(os.getenv("KEEP_ALIVE", "75"))

    # Optional HTTPS (direct access on bind)
    if tls_config:
        config.bind = [f"{host_https}:{ssl_port}"]
        config.certfile = tls_config["ssl_certfile"]
        config.keyfile = tls_config["ssl_keyfile"]
        logger.info("Binding HTTPS on %s:%s for direct access", host_https, ssl_port)
        logger.info("TrustyAI service running with dual HTTP/HTTPS protocol support")
    else:
        logger.info("TLS certificates not found - running HTTP only")

    # Configure logging
    config.accesslog = "-"  # Log to stdout
    config.errorlog = "-"  # Log to stderr
    config.use_reloader = False  # Disable reloader in production

    # Start the server
    # FastAPI implements the ASGI protocol that hypercorn expects
    # The type stubs are overly strict, but FastAPI works correctly at runtime
    await serve(app, config)  # type: ignore[arg-type]


if __name__ == "__main__":
    # SERVICE_STORAGE_FORMAT=PVC; STORAGE_DATA_FOLDER=/tmp; STORAGE_DATA_FILENAME=trustyai_test.hdf5
    asyncio.run(run_server())
