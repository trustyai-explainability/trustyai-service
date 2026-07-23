"""Identity metric endpoint for testing and validation purposes."""

import logging
from http import HTTPStatus
from typing import Never

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class IdentityMetricRequest(BaseModel):
    """Request parameters for identity metric computation."""

    modelId: str
    requestName: str | None = None
    metricName: str | None = None
    batchSize: int | None = 100
    columnName: str
    lowerThreshold: float | None = None
    upperThreshold: float | None = None


class ScheduleId(BaseModel):
    """Identifier for a scheduled metric computation request."""

    requestId: str


@router.post("/metrics/identity", response_model=None)
async def compute_identity_metric(request: IdentityMetricRequest) -> Never:
    """Provide a specific, plain-english interpretation of the current value of this metric."""
    logger.info(
        "Computing identity metric for model: %s, column: %s",
        request.modelId,
        request.columnName,
    )
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Identity metric computation is not yet implemented",
    )


@router.get("/metrics/identity/definition")
async def get_identity_definition() -> dict[str, str]:
    """Provide a general definition of this metric."""
    return {
        "name": "Identity Metric",
        "description": "",
    }


@router.post("/metrics/identity/definition", response_model=None)
async def interpret_identity_value(request: IdentityMetricRequest) -> Never:
    """Provide a specific, plain-english interpretation of a specific value of this metric."""
    logger.info("Interpreting identity metric value for model: %s", request.modelId)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Identity metric value interpretation is not yet implemented",
    )


@router.post("/metrics/identity/request", response_model=None)
async def schedule_identity(
    _request: IdentityMetricRequest, _background_tasks: BackgroundTasks
) -> Never:
    """Schedule a recurring computation of this metric."""
    logger.info("Scheduling identity metric computation")
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Identity metric scheduling is not yet implemented",
    )


@router.delete("/metrics/identity/request", response_model=None)
async def delete_identity_schedule(schedule: ScheduleId) -> Never:
    """Delete a recurring computation of this metric."""
    logger.info("Deleting identity schedule: %s", schedule.requestId)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Identity metric schedule deletion is not yet implemented",
    )


@router.get("/metrics/identity/requests", response_model=None)
async def list_identity_requests() -> Never:
    """List the currently scheduled computations of this metric."""
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Identity metric schedule listing is not yet implemented",
    )
