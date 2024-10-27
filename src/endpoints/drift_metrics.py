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
async def schedule_approxkstest(
    request: ApproxKSTestMetricRequest, background_tasks: BackgroundTasks
):
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
