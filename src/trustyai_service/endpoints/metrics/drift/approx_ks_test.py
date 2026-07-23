"""Approximate Kolmogorov-Smirnov test endpoint for drift detection."""

import logging
from http import HTTPStatus
from typing import Any, Never

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class ScheduleId(BaseModel):
    """Identifier for a scheduled metric computation request."""

    requestId: str


# ApproxKSTest
class GKSketch(BaseModel):
    """Greenwald-Khanna sketch parameters for approximate quantile estimation."""

    epsilon: float
    summary: list[dict[str, Any]] = []
    xmin: float
    xmax: float
    numx: int


class ApproxKSTestMetricRequest(BaseModel):
    """Request parameters for approximate Kolmogorov-Smirnov test metric computation."""

    modelId: str
    requestName: str | None = None
    metricName: str | None = None
    batchSize: int | None = 100
    thresholdDelta: float | None = None
    referenceTag: str | None = None
    fitColumns: list[str] = []
    epsilon: float | None = None
    sketchFitting: dict[str, GKSketch] | None = None


@router.post("/metrics/drift/approxkstest", response_model=None)
async def compute_approxkstest(request: ApproxKSTestMetricRequest) -> Never:
    """Compute the current value of ApproxKSTest metric."""
    logger.info("Computing ApproxKSTest for model: %s", request.modelId)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="ApproxKSTest metric computation is not yet implemented",
    )


@router.get("/metrics/drift/approxkstest/definition")
async def get_approxkstest_definition() -> dict[str, str]:
    """Provide a general definition of ApproxKSTest metric."""
    return {
        "name": "Approximate Kolmogorov-Smirnov Test",
        "description": "Description.",
    }


@router.post("/metrics/drift/approxkstest/request", response_model=None)
async def schedule_approxkstest(
    _request: ApproxKSTestMetricRequest, _background_tasks: BackgroundTasks
) -> Never:
    """Schedule a recurring computation of ApproxKSTest metric."""
    logger.info("Scheduling ApproxKSTest computation")
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="ApproxKSTest metric scheduling is not yet implemented",
    )


@router.delete("/metrics/drift/approxkstest/request", response_model=None)
async def delete_approxkstest_schedule(schedule: ScheduleId) -> Never:
    """Delete a recurring computation of ApproxKSTest metric."""
    logger.info("Deleting ApproxKSTest schedule: %s", schedule.requestId)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="ApproxKSTest metric schedule deletion is not yet implemented",
    )


@router.get("/metrics/drift/approxkstest/requests", response_model=None)
async def list_approxkstest_requests() -> Never:
    """List the currently scheduled computations of ApproxKSTest metric."""
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="ApproxKSTest metric schedule listing is not yet implemented",
    )
