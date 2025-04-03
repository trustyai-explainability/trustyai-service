from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)


class ScheduleId(BaseModel):
    requestId: str

# FourierMMD
class FourierMMDParameters(BaseModel):
    nWindow: Optional[int] = None
    nTest: Optional[int] = None
    nMode: Optional[int] = None
    randomSeed: Optional[int] = None
    sig: Optional[float] = None
    deltaStat: Optional[bool] = None
    epsilon: Optional[float] = None


class FourierMMDFitting(BaseModel):
    randomSeed: Optional[int] = None
    deltaStat: Optional[bool] = None
    nMode: Optional[int] = None
    scale: Optional[List[float]] = None
    aRef: Optional[List[float]] = None
    meanMMD: Optional[float] = None
    stdMMD: Optional[float] = None


class FourierMMDMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    thresholdDelta: Optional[float] = None
    referenceTag: Optional[str] = None
    fitColumns: List[str] = []
    parameters: Optional[FourierMMDParameters] = None
    gamma: Optional[float] = None
    fitting: Optional[FourierMMDFitting] = None


@router.post("/metrics/drift/fouriermmd")
async def compute_fouriermmd(request: FourierMMDMetricRequest):
    """Compute the current value of FourierMMD metric."""
    try:
        logger.info(f"Computing FourierMMD for model: {request.modelId}")
        # TODO: Implement
        return {"status": "success", "value": 0.5}
    except Exception as e:
        logger.error(f"Error computing FourierMMD: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/fouriermmd/definition")
async def get_fouriermmd_definition():
    """Provide a general definition of FourierMMD metric."""
    return {
        "name": "FourierMMD Drift",
        "description": "Description",
    }


@router.post("/metrics/drift/fouriermmd/request")
async def schedule_fouriermmd(
    request: FourierMMDMetricRequest, background_tasks: BackgroundTasks
):
    """Schedule a recurring computation of FourierMMD metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling FourierMMD computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/drift/fouriermmd/request")
async def delete_fouriermmd_schedule(schedule: ScheduleId):
    """Delete a recurring computation of FourierMMD metric."""
    logger.info(f"Deleting FourierMMD schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/drift/fouriermmd/requests")
async def list_fouriermmd_requests():
    """List the currently scheduled computations of FourierMMD metric."""
    # TODO: Implement
    return {"requests": []}