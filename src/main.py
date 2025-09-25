import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# from fastapi_utils.tasks import repeat_every  # Removed due to compatibility issues
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Endpoint routers
from src.endpoints.consumer.consumer_endpoint import router as consumer_router
from src.endpoints.data.data_download import router as data_download_router
from src.endpoints.data.data_upload import router as data_upload_router

# from src.endpoints.explainers import router as explainers_router
from src.endpoints.explainers.global_explainer import router as explainers_global_router
from src.endpoints.explainers.local_explainer import router as explainers_local_router
from src.endpoints.metadata import router as metadata_router

# from src.endpoints.drift_metrics import router as drift_metrics_router
from src.endpoints.metrics.drift.approx_ks_test import (
    router as drift_approx_ks_test_router,
)
from src.endpoints.metrics.drift.fourier_mmd import router as drift_fourier_mmd_router
from src.endpoints.metrics.drift.ks_test import router as drift_ks_test_router
from src.endpoints.metrics.drift.meanshift import router as drift_meanshift_router
from src.endpoints.metrics.fairness.group.dir import router as dir_router
from src.endpoints.metrics.fairness.group.spd import router as spd_router
from src.endpoints.metrics.identity.identity_endpoint import router as identity_router
from src.endpoints.metrics.metrics_info import router as metrics_info_router

from src.service.prometheus.prometheus_scheduler import PrometheusScheduler

try:
    from src.endpoints.evaluation.lm_evaluation_harness import (
        router as lm_evaluation_harness_router,
    )

    lm_evaluation_harness_available = True
except ImportError:
    lm_evaluation_harness_available = False

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

prometheus_scheduler = PrometheusScheduler()


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
# app.include_router(
#     drift_metrics_router,
#     tags=[
#         "Drift Metrics: ApproxKSTest",
#         "Drift Metrics: FourierMMD Drift",
#         "Drift Metrics: KSTest",
#         "Drift Metrics: Meanshift",
#     ],
# )
app.include_router(
    drift_approx_ks_test_router,
    tags=[
        "Drift Metrics: ApproxKSTest",
    ],
)
app.include_router(
    drift_fourier_mmd_router,
    tags=[
        "Drift Metrics: FourierMMD Drift",
    ],
)
app.include_router(
    drift_ks_test_router,
    tags=[
        "Drift Metrics: KSTest",
    ],
)
app.include_router(
    drift_meanshift_router,
    tags=[
        "Drift Metrics: Meanshift",
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
app.include_router(data_download_router, tags=["Download Endpoint"])

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


if __name__ == "__main__":
    # SERVICE_STORAGE_FORMAT=PVC; STORAGE_DATA_FOLDER=/tmp; STORAGE_DATA_FILENAME=trustyai_test.hdf5

    # Get TLS configuration
    tls_config = get_tls_config()

    # Configure server settings
    host = "0.0.0.0"
    http_port = int(os.getenv("HTTP_PORT", "8080"))
    ssl_port = int(os.getenv("SSL_PORT", "4443"))

    if tls_config:
        # Run dual servers: HTTP for health checks, HTTPS for main service
        import threading
        import time

        def run_http_server():
            logger.info(f"Starting TrustyAI HTTP server for health checks on port {http_port}")
            uvicorn.run(app=app, host=host, port=http_port, log_level="warning")

        def run_https_server():
            logger.info(f"Starting TrustyAI HTTPS server on port {ssl_port}")
            uvicorn.run(app=app, host=host, port=ssl_port, **tls_config)

        # Start HTTP server in background thread for health checks
        http_thread = threading.Thread(target=run_http_server, daemon=True)
        http_thread.start()

        # Give HTTP server time to start
        time.sleep(1)

        # Run HTTPS server in main thread
        run_https_server()
    else:
        # Run without TLS on HTTP port
        logger.info(f"Starting TrustyAI service without TLS on port {http_port}")
        uvicorn.run(app=app, host=host, port=http_port)
