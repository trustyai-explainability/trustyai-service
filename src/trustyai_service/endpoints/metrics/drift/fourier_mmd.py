"""Fourier Maximum Mean Discrepancy (MMD) endpoint for drift detection."""

import logging
from http import HTTPStatus
from typing import Never

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class ScheduleId(BaseModel):
    """Identifier for a scheduled metric computation request."""

    requestId: str


# FourierMMD
class FourierMMDParameters(BaseModel):
    """Parameters for Fourier MMD drift detection algorithm."""

    nWindow: int | None = None
    nTest: int | None = None
    nMode: int | None = None
    randomSeed: int | None = None
    sig: float | None = None
    deltaStat: bool | None = None
    epsilon: float | None = None


class FourierMMDFitting(BaseModel):
    """Fitted reference distribution parameters for Fourier MMD."""

    randomSeed: int | None = None
    deltaStat: bool | None = None
    nMode: int | None = None
    scale: list[float] | None = None
    aRef: list[float] | None = None
    meanMMD: float | None = None
    stdMMD: float | None = None


class FourierMMDMetricRequest(BaseModel):
    """Request parameters for Fourier MMD drift detection metric computation."""

    modelId: str
    requestName: str | None = None
    metricName: str | None = None
    batchSize: int | None = 100
    thresholdDelta: float | None = None
    referenceTag: str | None = None
    fitColumns: list[str] = []
    parameters: FourierMMDParameters | None = None
    gamma: float | None = None
    fitting: FourierMMDFitting | None = None


@router.post("/metrics/drift/fouriermmd", response_model=None)
async def compute_fouriermmd(request: FourierMMDMetricRequest) -> Never:
    """Compute the current value of FourierMMD metric."""
    logger.info("Computing FourierMMD for model: %s", request.modelId)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="FourierMMD metric computation is not yet implemented",
    )


@router.get("/metrics/drift/fouriermmd/definition")
async def get_fouriermmd_definition() -> dict[str, str]:
    """Provide a general definition of FourierMMD metric."""
    return {
        "name": "FourierMMD Drift",
        "description": "Description",
    }


@router.post("/metrics/drift/fouriermmd/request", response_model=None)
async def schedule_fouriermmd(
    _request: FourierMMDMetricRequest, _background_tasks: BackgroundTasks
) -> Never:
    """Schedule a recurring computation of FourierMMD metric."""
    logger.info("Scheduling FourierMMD computation")
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="FourierMMD metric scheduling is not yet implemented",
    )


@router.delete("/metrics/drift/fouriermmd/request", response_model=None)
async def delete_fouriermmd_schedule(schedule: ScheduleId) -> Never:
    """Delete a recurring computation of FourierMMD metric."""
    logger.info("Deleting FourierMMD schedule: %s", schedule.requestId)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="FourierMMD metric schedule deletion is not yet implemented",
    )


@router.get("/metrics/drift/fouriermmd/requests", response_model=None)
async def list_fouriermmd_requests() -> Never:
    """List the currently scheduled computations of FourierMMD metric."""
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="FourierMMD metric schedule listing is not yet implemented",
    )
