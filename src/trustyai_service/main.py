"""TrustyAI service main application entry point and FastAPI configuration."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hypercorn.asyncio import serve
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from trustyai_service.endpoints import routes

# Endpoint routers
from trustyai_service.endpoints.consumer.consumer_endpoint import (
    consume_cloud_event,
    consume_inference_payload,
)
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
from trustyai_service.endpoints.metrics.drift.jensen_shannon import (
    router as drift_jensenshannon_router,
)
from trustyai_service.endpoints.metrics.drift.kolmogorov_smirnov import (
    router as drift_kstest_router,
)
from trustyai_service.endpoints.metrics.drift.kolmogorov_smirnov_streaming import (
    router as drift_ksteststreaming_router,
)
from trustyai_service.endpoints.metrics.drift.mmd import router as drift_mmd_router
from trustyai_service.endpoints.metrics.fairness.group.dir import router as dir_router
from trustyai_service.endpoints.metrics.fairness.group.spd import router as spd_router
from trustyai_service.endpoints.metrics.metrics_info import (
    router as metrics_info_router,
)

# Middleware
from trustyai_service.middleware.gzip_middleware import GzipRequestMiddleware

# Feature flag gating
from trustyai_service.service.config.registry import (
    register_if_enabled,
    register_if_enabled_with_group,
    register_with_legacy_prefix,
)

# Health checks
from trustyai_service.service.health_checks import (
    STATUS_OK,
    perform_liveness_checks,
    perform_readiness_checks,
)
from trustyai_service.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)
from trustyai_service.service.tls import PolicyAwareConfig

logging.basicConfig(
    level=logging.INFO,  # Reduce default verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Enable debug logging for TrustyAI components only
logging.getLogger("src").setLevel(logging.DEBUG)
logging.getLogger("__main__").setLevel(logging.DEBUG)

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


# Include core routers (always registered)
app.include_router(
    consumer_router,
    tags=["{Internal Only} Inference Consumer", "{Internal Only} ModelMesh Consumer"],
)
app.include_router(data_upload_router, tags=["Data Upload"])
app.include_router(batch_mean_router, tags=["Metrics: Batch Mean"])
app.include_router(metadata_router, tags=["Service Metadata"])
app.include_router(metrics_info_router, tags=["Metrics Information Endpoint"])

# Fairness metrics (feature-flag gated, with legacy /metrics prefix)
register_with_legacy_prefix(
    app,
    dir_router,
    "fairness",
    "fairness_dir",
    modern_tag="Fairness Metrics: Group: Disparate Impact Ratio",
    legacy_tag="{Legacy}: Disparate Impact Ratio",
)
register_with_legacy_prefix(
    app,
    spd_router,
    "fairness",
    "fairness_spd",
    modern_tag="Fairness Metrics: Group: Statistical Parity Difference",
    legacy_tag="{Legacy}: Statistical Parity Difference",
)

# Drift metrics (feature-flag gated)
register_if_enabled_with_group(
    app,
    drift_comparemeans_router,
    "drift",
    "drift_compare_means",
    tag="Drift Metrics: CompareMeans",
)
register_if_enabled_with_group(
    app,
    drift_mmd_router,
    "drift",
    "drift_mmd",
    tag="Drift Metrics: MMD",
)
register_if_enabled_with_group(
    app,
    drift_jensenshannon_router,
    "drift",
    "drift_jensen_shannon",
    tag="Drift Metrics: JensenShannon",
)
register_if_enabled_with_group(
    app,
    drift_kstest_router,
    "drift",
    "drift_ks_test",
    tag="Drift Metrics: KSTest",
)
# KSTestStreaming doesn't have its own flag yet, gate with drift group
register_if_enabled(
    app,
    drift_ksteststreaming_router,
    "drift",
    tag="Drift Metrics: KSTestStreaming",
)

# Explainer endpoints (feature-flag gated, disabled by default)
register_if_enabled_with_group(
    app,
    explainers_global_router,
    "explainer",
    "explainer_global",
    tag="Explainers: Global",
)
register_if_enabled_with_group(
    app,
    explainers_local_router,
    "explainer",
    "explainer_local",
    tag="Explainers: Local",
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning service welcome message.

    :return: Dictionary with welcome message
    """
    return {"message": "Welcome to TrustyAI Explainability Service"}


@app.get(routes.PROMETHEUS_METRICS)
async def metrics(_request: Request) -> Response:
    """Prometheus metrics endpoint.

    :param _request: FastAPI request object (unused)
    :return: Prometheus metrics in text format
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(routes.HEALTH)
def general_health() -> JSONResponse:
    """General health endpoint combining readiness and liveness checks.

    :return: JSON response with status ("healthy" or "unhealthy")
             HTTP 200 if healthy, HTTP 503 if unhealthy
    """
    readiness_status, readiness_checks = perform_readiness_checks()
    liveness_status, liveness_checks = perform_liveness_checks()

    is_healthy = readiness_status == STATUS_OK and liveness_status == STATUS_OK

    response_body = {
        "status": "healthy" if is_healthy else "unhealthy",
        "checks": {
            "readiness": readiness_checks,
            "liveness": liveness_checks,
        },
    }

    status_code = HTTPStatus.OK if is_healthy else HTTPStatus.SERVICE_UNAVAILABLE
    return JSONResponse(content=response_body, status_code=status_code)


# Readiness probe
@app.get(routes.HEALTH_READY)
def readiness_probe() -> JSONResponse:
    """Kubernetes readiness probe endpoint.

    :return: JSON response with status ("ready" or "not_ready")
             HTTP 200 if ready, HTTP 503 if not ready
    """
    status, checks = perform_readiness_checks()
    is_ready = status == STATUS_OK

    response_body = {"status": "ready" if is_ready else "not_ready", "checks": checks}

    status_code = HTTPStatus.OK if is_ready else HTTPStatus.SERVICE_UNAVAILABLE
    return JSONResponse(content=response_body, status_code=status_code)


# Liveness probe endpoint
@app.get(routes.HEALTH_LIVE)
def liveness_probe() -> JSONResponse:
    """Kubernetes liveness probe endpoint.

    :return: JSON response with status ("alive")
             HTTP 200 if alive
    """
    status, checks = perform_liveness_checks()
    is_alive = status == STATUS_OK

    response_body = {"status": "alive" if is_alive else "dead", "checks": checks}

    status_code = HTTPStatus.OK if is_alive else HTTPStatus.SERVICE_UNAVAILABLE
    return JSONResponse(content=response_body, status_code=status_code)


health_app = FastAPI(
    title="TrustyAI Health",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
health_app.get(routes.HEALTH)(general_health)
health_app.get(routes.HEALTH_READY)(readiness_probe)
health_app.get(routes.HEALTH_LIVE)(liveness_probe)

# KServe consumer endpoints on the health port (8080).
# The ModelMesh agent sends flat payloads to /consumer/kserve/v2.
# The KServe inference logger sends CloudEvents to /.
health_app.post(routes.CONSUMER_KSERVE_V2)(consume_inference_payload)
health_app.post(routes.CONSUMER_ROOT)(consume_cloud_event)


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
    http_port = int(os.getenv("HTTP_PORT", "8081"))
    ssl_port = int(os.getenv("SSL_PORT", "4443"))
    health_port = int(os.getenv("HEALTH_PORT", "8080"))

    if health_port in (http_port, ssl_port):
        msg = f"HEALTH_PORT ({health_port}) must differ from HTTP_PORT ({http_port}) and SSL_PORT ({ssl_port})"
        raise ValueError(msg)

    if http_port == ssl_port:
        msg = f"HTTP_PORT ({http_port}) must differ from SSL_PORT ({ssl_port})"
        raise ValueError(msg)

    # Create hypercorn config
    config = PolicyAwareConfig()

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

    health_config = PolicyAwareConfig()
    health_config.bind = [f"0.0.0.0:{health_port}"]
    health_config.use_reloader = False
    logger.info("Binding health probes on 0.0.0.0:%s for kubelet", health_port)

    await asyncio.gather(
        serve(app, config),  # type: ignore[arg-type]
        serve(health_app, health_config),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    # SERVICE_STORAGE_FORMAT=PVC; STORAGE_DATA_FOLDER=/tmp; STORAGE_DATA_FILENAME=trustyai_test.hdf5
    asyncio.run(run_server())
