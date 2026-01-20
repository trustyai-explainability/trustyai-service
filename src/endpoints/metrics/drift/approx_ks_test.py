from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)


class ScheduleId(BaseModel):
    requestId: str


# ApproxKSTest
class GKSketch(BaseModel):
    epsilon: float
    summary: List[Dict[str, Any]] = []
    xmin: float
    xmax: float
    numx: int


class ApproxKSTestMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    thresholdDelta: Optional[float] = None
    referenceTag: Optional[str] = None
    fitColumns: List[str] = []
    epsilon: Optional[float] = None
    sketchFitting: Optional[Dict[str, GKSketch]] = None


@router.post("/metrics/drift/approxkstest")
async def compute_approxkstest(request: ApproxKSTestMetricRequest):
    """Compute the current value of ApproxKSTest metric."""
    try:
        logger.info(f"Computing ApproxKSTest for model: {request.modelId}")
        # TODO: Implement
        return {"status": "success", "value": 0.5}
    except Exception as e:
        logger.error(f"Error computing ApproxKSTest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/approxkstest/definition")
async def get_approxkstest_definition():
    """Provide a general definition of ApproxKSTest metric."""
    return {
        "name": "Approximate Kolmogorov-Smirnov Test",
        "description": "Description.",
    }


@router.post("/metrics/drift/approxkstest/request")
async def schedule_approxkstest(request: ApproxKSTestMetricRequest, background_tasks: BackgroundTasks):
    """Schedule a recurring computation of ApproxKSTest metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling ApproxKSTest computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/drift/approxkstest/request")
async def delete_approxkstest_schedule(schedule: ScheduleId):
    """Delete a recurring computation of ApproxKSTest metric."""
    logger.info(f"Deleting ApproxKSTest schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/drift/approxkstest/requests")
async def list_approxkstest_requests():
    """List the currently scheduled computations of ApproxKSTest metric."""
    # TODO: Implement
    return {"requests": []}
