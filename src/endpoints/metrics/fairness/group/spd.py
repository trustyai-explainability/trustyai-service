from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)


class ReconcilableFeature(BaseModel):
    rawValueNodes: Optional[List[Dict[str, Any]]] = None
    rawValueNode: Optional[Dict[str, Any]] = None
    reconciledType: Optional[List[Dict[str, Any]]] = None
    multipleValued: Optional[bool] = None


class ReconcilableOutput(BaseModel):
    rawValueNodes: Optional[List[Dict[str, Any]]] = None
    rawValueNode: Optional[Dict[str, Any]] = None
    reconciledType: Optional[List[Dict[str, Any]]] = None
    multipleValued: Optional[bool] = None


class GroupMetricRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    protectedAttribute: str
    outcomeName: str
    privilegedAttribute: ReconcilableFeature
    unprivilegedAttribute: ReconcilableFeature
    favorableOutcome: ReconcilableOutput
    thresholdDelta: Optional[float] = None


class GroupDefinitionRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    protectedAttribute: str
    outcomeName: str
    privilegedAttribute: ReconcilableFeature
    unprivilegedAttribute: ReconcilableFeature
    favorableOutcome: ReconcilableOutput
    thresholdDelta: Optional[float] = None
    metricValue: Dict[str, Any]


class ScheduleId(BaseModel):
    requestId: str


# Statistical Parity Difference
@router.post("/metrics/group/fairness/spd")
async def compute_spd(request: GroupMetricRequest):
    """Compute the current value of Statistical Parity Difference metric."""
    try:
        logger.info(f"Computing SPD for model: {request.modelId}")
        # TODO: Implement
        return {"status": "success", "value": 0.1}
    except Exception as e:
        logger.error(f"Error computing SPD: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/group/fairness/spd/definition")
async def get_spd_definition():
    """Provide a general definition of Statistical Parity Difference metric."""
    return {
        "name": "Statistical Parity Difference",
        "description": "Description.",
    }


@router.post("/metrics/group/fairness/spd/definition")
async def interpret_spd_value(request: GroupDefinitionRequest):
    """Provide a specific, plain-english interpretation of a specific value of SPD metric."""
    try:
        logger.info(f"Interpreting SPD value for model: {request.modelId}")
        # TODO: Implement
        return {"interpretation": "The SPD value indicates a small bias in the model."}
    except Exception as e:
        logger.error(f"Error interpreting SPD value: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error interpreting value: {str(e)}"
        )


@router.post("/metrics/group/fairness/spd/request")
async def schedule_spd(request: GroupMetricRequest, background_tasks: BackgroundTasks):
    """Schedule a recurring computation of SPD metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling SPD computation with ID: {request_id}")
    # TODO: Implement
    return {"requestId": request_id}


@router.delete("/metrics/group/fairness/spd/request")
async def delete_spd_schedule(schedule: ScheduleId):
    """Delete a recurring computation of SPD metric."""
    logger.info(f"Deleting SPD schedule: {schedule.requestId}")
    # TODO: Implement
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/group/fairness/spd/requests")
async def list_spd_requests():
    """List the currently scheduled computations of SPD metric."""
    # TODO: Implement
    return {"requests": []}


# Deprecated SPD endpoints
@router.post("/spd", deprecated=True)
async def compute_spd_deprecated(request: GroupMetricRequest):
    """Compute the current value of Statistical Parity Difference metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd instead.
    """
    return await compute_spd(request)


@router.get("/spd/definition", deprecated=True)
async def get_spd_definition_deprecated():
    """Provide a general definition of Statistical Parity Difference metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/definition instead.
    """
    return await get_spd_definition()


@router.post("/spd/definition", deprecated=True)
async def interpret_spd_value_deprecated(request: GroupDefinitionRequest):
    """Provide a specific interpretation of a SPD metric value (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/definition instead.
    """
    return await interpret_spd_value(request)


@router.post("/spd/request", deprecated=True)
async def schedule_spd_deprecated(
    request: GroupMetricRequest, background_tasks: BackgroundTasks
):
    """Schedule a recurring computation of SPD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/request instead.
    """
    return await schedule_spd(request, background_tasks)


@router.delete("/spd/request", deprecated=True)
async def delete_spd_schedule_deprecated(schedule: ScheduleId):
    """Delete a recurring computation of SPD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/request instead.
    """
    return await delete_spd_schedule(schedule)


@router.get("/spd/requests", deprecated=True)
async def list_spd_requests_deprecated():
    """List the currently scheduled computations of SPD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/requests instead.
    """
    return await list_spd_requests()
