import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config

# from fastapi_utils.tasks import repeat_every  # Removed due to compatibility issues
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Endpoint routers
from src.endpoints.consumer.consumer_endpoint import router as consumer_router
from src.endpoints.data.data_upload import router as data_upload_router

# from src.endpoints.explainers import router as explainers_router
from src.endpoints.explainers.global_explainer import router as explainers_global_router
from src.endpoints.explainers.local_explainer import router as explainers_local_router
from src.endpoints.metadata import router as metadata_router

# from src.endpoints.drift_metrics import router as drift_metrics_router
from src.endpoints.metrics.drift.kolmogorov_smirnov import router as drift_kstest_router
from src.endpoints.metrics.fairness.group.dir import router as dir_router
from src.endpoints.metrics.fairness.group.spd import router as spd_router
from src.endpoints.metrics.identity.identity_endpoint import router as identity_router
from src.endpoints.metrics.metrics_info import router as metrics_info_router
from src.service.prometheus.shared_prometheus_scheduler import get_shared_prometheus_scheduler

try:
    from src.endpoints.evaluation.lm_evaluation_harness import (
        router as lm_evaluation_harness_router,
    )

    lm_evaluation_harness_available = True
except ImportError:
    lm_evaluation_harness_available = False

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
scheduler_logger = logging.getLogger("src.service.prometheus.prometheus_scheduler")
scheduler_logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

prometheus_scheduler = get_shared_prometheus_scheduler()


async def schedule_metrics_calculation():
    """Background task to calculate metrics at regular intervals."""
    while True:
        try:
            await prometheus_scheduler.calculate()
        except Exception as e:
            logger.error(f"Error in metrics calculation: {e}")

        # Wait for the configured interval
        interval = prometheus_scheduler.service_config.get("metrics_schedule", 30)
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(
    consumer_router,
    tags=["{Internal Only} Inference Consumer", "{Internal Only} ModelMesh Consumer"],
)
app.include_router(dir_router, tags=["Fairness Metrics: Group: Disparate Impact Ratio"])
app.include_router(data_upload_router, tags=["Data Upload"])

#   Drift metrics
app.include_router(
    drift_kstest_router,
    tags=[
        "Drift Metrics: KSTest",
    ],
)

# app.include_router(explainers_router, tags=["Explainers: Global", "Explainers: Local"])
app.include_router(explainers_global_router, tags=["Explainers: Global"])
app.include_router(explainers_local_router, tags=["Explainers: Local"])
app.include_router(
    spd_router,
    tags=["Fairness Metrics: Group: Statistical Parity Difference"],
)
app.include_router(identity_router, tags=["Identity Endpoint"])
app.include_router(metadata_router, tags=["Service Metadata"])
app.include_router(metrics_info_router, tags=["Metrics Information Endpoint"])

if lm_evaluation_harness_available:
    app.include_router(lm_evaluation_harness_router, tags=["LM Evaluation Harness Endpoint"])

# Deprecated endpoints
app.include_router(dir_router, prefix="/metrics", tags=["{Legacy}: Disparate Impact Ratio"])
app.include_router(
    spd_router,
    prefix="/metrics",
    tags=["{Legacy}: Statistical Parity Difference"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to TrustyAI Explainability Service"}


@app.get("/q/metrics")
async def metrics(request: Request):
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Readiness probe
@app.get("/q/health/ready")
async def readiness_probe():
    return JSONResponse(content={"status": "ready"}, status_code=200)


# Liveness probe endpoint
@app.get("/q/health/live")
async def liveness_probe():
    return JSONResponse(content={"status": "live"}, status_code=200)


def get_tls_config():
    """
    Get TLS configuration for the service.
    Returns SSL configuration if certificates are available, None otherwise.
    """
    cert_file = os.getenv("TLS_CERT_FILE", "/etc/tls/internal/tls.crt")
    key_file = os.getenv("TLS_KEY_FILE", "/etc/tls/internal/tls.key")

    cert_path = Path(cert_file)
    key_path = Path(key_file)

    if cert_path.exists() and key_path.exists():
        logger.info(f"TLS certificates found at {cert_file} and {key_file}")
        return {
            "ssl_keyfile": str(key_path),
            "ssl_certfile": str(cert_path),
            "ssl_version": 2,  # TLS v1.2+
        }
    else:
        logger.info("TLS certificates not found, running in HTTP mode")
        return None


async def run_server():
    """Run hypercorn server with both HTTP and HTTPS binds"""
    # Get TLS configuration
    tls_config = get_tls_config()

    # Configure server settings
    host_https = "0.0.0.0"
    host_http = "127.0.0.1"  # Keep loopback-only for security (kube-rbac-proxy forwards here)
    http_port = int(os.getenv("HTTP_PORT", "8080"))
    ssl_port = int(os.getenv("SSL_PORT", "4443"))

    # Create hypercorn config
    config = Config()

    # HTTP for kube-rbac-proxy (plain HTTP on insecure_bind)
    config.insecure_bind = [f"{host_http}:{http_port}"]
    logger.info(f"Binding HTTP on {host_http}:{http_port} for kube-rbac-proxy")

    # Configure for HTTP/1.1 compatibility and proper keep-alive
    config.h11_max_incomplete_size = 16 * 1024 * 1024  # 16MB for large requests
    config.keep_alive = int(os.getenv("KEEP_ALIVE", "75"))  # Allow override via env var

    # Optional HTTPS (direct access on bind)
    if tls_config:
        config.bind = [f"{host_https}:{ssl_port}"]
        config.certfile = tls_config["ssl_certfile"]
        config.keyfile = tls_config["ssl_keyfile"]
        logger.info(f"Binding HTTPS on {host_https}:{ssl_port} for direct access")
        logger.info("TrustyAI service running with dual HTTP/HTTPS protocol support")
    else:
        logger.info("TLS certificates not found - running HTTP only")

    # Configure logging
    config.accesslog = "-"  # Log to stdout
    config.errorlog = "-"  # Log to stderr
    config.use_reloader = False  # Disable reloader in production

    # Start the server
    await serve(app, config)


if __name__ == "__main__":
    # SERVICE_STORAGE_FORMAT=PVC; STORAGE_DATA_FOLDER=/tmp; STORAGE_DATA_FILENAME=trustyai_test.hdf5
    asyncio.run(run_server())
