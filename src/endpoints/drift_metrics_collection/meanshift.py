from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)


class ScheduleId(BaseModel):
    requestId: str


# Meanshift
class StatisticalSummaryValues(BaseModel):
    mean: float
    variance: float
    n: int
    max: float
    min: float
    sum: float
    standardDeviation: float


class MeanshiftMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    thresholdDelta: Optional[float] = None
    referenceTag: Optional[str] = None
    fitColumns: List[str] = []
    fitting: Optional[Dict[str, StatisticalSummaryValues]] = None


@router.post("/metrics/drift/meanshift")
async def compute_meanshift(request: MeanshiftMetricRequest):
    """Compute the current value of Meanshift metric."""
    try:
        logger.info(f"Computing Meanshift for model: {request.modelId}")
        # TODO: Implement
        return {"status": "success", "value": 0.5}
    except Exception as e:
        logger.error(f"Error computing Meanshift: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/meanshift/definition")
async def get_meanshift_definition():
    """Provide a general definition of Meanshift metric."""
    return {
        "name": "Meanshift",
        "description": "Description.",
    }


@router.post("/metrics/drift/meanshift/request")
async def schedule_meanshift(
    request: MeanshiftMetricRequest, background_tasks: BackgroundTasks
):
    """Schedule a recurring computation of Meanshift metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling Meanshift computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/drift/meanshift/request")
async def delete_meanshift_schedule(schedule: ScheduleId):
    """Delete a recurring computation of Meanshift metric."""
    logger.info(f"Deleting Meanshift schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/drift/meanshift/requests")
async def list_meanshift_requests():
    """List the currently scheduled computations of Meanshift metric."""
    # TODO: Implement
    return {"requests": []}
