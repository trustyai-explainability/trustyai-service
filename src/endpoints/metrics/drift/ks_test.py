from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)


class ScheduleId(BaseModel):
    requestId: str


# KSTest
class KSTestMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    thresholdDelta: Optional[float] = None
    referenceTag: Optional[str] = None
    fitColumns: List[str] = []


@router.post("/metrics/drift/kstest")
async def compute_kstest(request: KSTestMetricRequest):
    """Compute the current value of KSTest metric."""
    try:
        logger.info(f"Computing KSTest for model: {request.modelId}")
        # TODO: Implement
        return {"status": "success", "value": 0.5}
    except Exception as e:
        logger.error(f"Error computing KSTest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/kstest/definition")
async def get_kstest_definition():
    """Provide a general definition of KSTest metric."""
    return {
        "name": "Kolmogorov-Smirnov Test",
        "description": "Description.",
    }


@router.post("/metrics/drift/kstest/request")
async def schedule_kstest(
    request: KSTestMetricRequest, background_tasks: BackgroundTasks
):
    """Schedule a recurring computation of KSTest metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling KSTest computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/drift/kstest/request")
async def delete_kstest_schedule(schedule: ScheduleId):
    """Delete a recurring computation of KSTest metric."""
    logger.info(f"Deleting KSTest schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/drift/kstest/requests")
async def list_kstest_requests():
    """List the currently scheduled computations of KSTest metric."""
    # TODO: Implement
    return {"requests": []}
