import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_utils.tasks import repeat_every
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


@repeat_every(
    seconds=prometheus_scheduler.service_config.get("metrics_schedule", 30),
    logger=logger,
    raise_exceptions=False,
)
async def schedule_metrics_calculation():
    await prometheus_scheduler.calculate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(schedule_metrics_calculation())

    yield

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


if __name__ == "__main__":
    # SERVICE_STORAGE_FORMAT=PVC; STORAGE_DATA_FOLDER=/tmp; STORAGE_DATA_FILENAME=trustyai_test.hdf5
    uvicorn.run(app=app, host="0.0.0.0", port=8080)
