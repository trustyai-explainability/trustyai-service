from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from prometheus_client import Gauge
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import pandas as pd
import os
import time
import uuid
import logging

from src.service.payloads.metrics.base_metric_request import BaseMetricRequest

router = APIRouter()
logger = logging.getLogger(__name__)

DATA_DIR = "data"


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


class GroupMetricRequest(BaseMetricRequest):
    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_id: str = Field(alias="modelId")
    metric_name: Optional[str] = Field(default=None, alias="metricName")  # Will be set by endpoint
    request_name: Optional[str] = Field(default=None, alias="requestName")
    batch_size: Optional[int] = Field(default=100, alias="batchSize")

    # DIR-specific fields
    protected_attribute: str = Field(alias="protectedAttribute")
    outcome_name: str = Field(alias="outcomeName")
    privileged_attribute: Union[ReconcilableFeature, int, float, str] = Field(alias="privilegedAttribute")
    unprivileged_attribute: Union[ReconcilableFeature, int, float, str] = Field(alias="unprivilegedAttribute")
    favorable_outcome: Union[ReconcilableOutput, int, float, str] = Field(alias="favorableOutcome")
    threshold_delta: Optional[float] = Field(default=None, alias="thresholdDelta")

    def retrieve_tags(self) -> Dict[str, str]:
        """Retrieve tags for this DIR metric request."""
        tags = self.retrieve_default_tags()
        tags["protectedAttribute"] = self.protected_attribute
        tags["outcomeName"] = self.outcome_name
        return tags


class GroupDefinitionRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    protectedAttribute: str
    outcomeName: str
    privilegedAttribute: Union[ReconcilableFeature, int, float, str]
    unprivilegedAttribute: Union[ReconcilableFeature, int, float, str]
    favorableOutcome: Union[ReconcilableOutput, int, float, str]
    thresholdDelta: Optional[float] = None
    metricValue: Dict[str, Any]


class ScheduleId(BaseModel):
    requestId: str


class MetricValueCarrier(BaseModel):
    value: float


def load_dataframe(model_name: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR, f"{model_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data for model '{model_name}' not found.")
    return pd.read_csv(file_path)


def calculate_disparate_impact_ratio(
    df: pd.DataFrame,
    protected_attr: str,
    outcome_name: str,
    favorable_outcome: List[str],
    privileged: List[str],
    unprivileged: List[str],
) -> float:
    privileged_group = df[df[protected_attr].isin(privileged)]
    unprivileged_group = df[df[protected_attr].isin(unprivileged)]

    priv_outcome_ratio = len(privileged_group[privileged_group[outcome_name].isin(favorable_outcome)]) / len(
        privileged_group
    )
    unpriv_outcome_ratio = len(unprivileged_group[unprivileged_group[outcome_name].isin(favorable_outcome)]) / len(
        unprivileged_group
    )

    if priv_outcome_ratio == 0:
        return 0.0
    return unpriv_outcome_ratio / priv_outcome_ratio


@router.post("/metrics/group/fairness/dir", response_model=MetricValueCarrier)
async def get_disparate_impact_ratio(request: GroupMetricRequest, delta: Optional[float] = Query(None)):
    try:
        df = load_dataframe(request.modelId)
        dir_value = calculate_disparate_impact_ratio(
            df,
            request.protectedAttribute,
            request.outcomeName,
            request.favorableOutcome.rawValueNodes,
            request.privilegedAttribute.rawValueNodes,
            request.unprivilegedAttribute.rawValueNodes,
        )
        return MetricValueCarrier(value=dir_value)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating DIR: {str(e)}")


class DIRRequest(BaseModel):
    modelid: str
    protected_attribute: str
    privileged_attribute: Union[str, float]
    unprivileged_attribute: Union[str, float]
    outcome_name: str
    favorable_outcome: List[str]
    batch_size: int


# Dictionary to store DIR requests
active_dir_requests = {}
dir_metric_gauge = Gauge("trustyai_dir", "Stored DIR Metric Values", ["request_id"])


def calculate_and_update_dir_metric(request_id: str):
    while request_id in active_dir_requests:
        request = active_dir_requests[request_id]

        try:
            df = load_dataframe(request.modelid)
            dir_value = calculate_disparate_impact_ratio(
                df,
                request.protected_attribute,
                request.outcome_name,
                request.favorable_outcome,
                [request.privileged_attribute],
                [request.unprivileged_attribute],
            )
            dir_metric_gauge.labels(request_id=request_id).set(dir_value)
            time.sleep(30)
        except Exception as e:
            print(f"Error calculating DIR for request {request_id}: {e}")
            break


@router.post("/metrics/group/fairness/dir/request")
async def register_dir_request(request: DIRRequest, background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    active_dir_requests[request_id] = request
    background_tasks.add_task(calculate_and_update_dir_metric, request_id)
    return {"request_id": request_id}


@router.delete("/metrics/group/fairness/dir/request/{request_id}")
async def delete_dir_request(request_id: str):
    if request_id in active_dir_requests:
        del active_dir_requests[request_id]
        dir_metric_gauge.remove(request_id=request_id)
        return {"detail": f"Request {request_id} has been deleted."}
    else:
        raise HTTPException(status_code=404, detail="Request ID not found.")


# Disparate Impact Ratio
@router.post("/metrics/group/fairness/dir")
async def compute_dir(request: GroupMetricRequest):
    """Compute the current value of Disparate Impact Ratio metric."""
    try:
        logger.info(f"Computing DIR for model: {request.modelId}")
        # TODO: Implement
        return {"status": "success", "value": 0.8}
    except Exception as e:
        logger.error(f"Error computing DIR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/group/fairness/dir/definition")
async def get_dir_definition():
    """Provide a general definition of Disparate Impact Ratio metric."""
    return {
        "name": "Disparate Impact Ratio",
        "description": "Description",
    }


@router.post("/metrics/group/fairness/dir/definition")
async def interpret_dir_value(request: GroupDefinitionRequest):
    """Provide a specific, plain-english interpretation of a specific value of DIR metric."""
    try:
        logger.info(f"Interpreting DIR value for model: {request.modelId}")
        # TODO: Implement interpretation
        return {"interpretation": "The DIR value..."}
    except Exception as e:
        logger.error(f"Error interpreting DIR value: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interpreting value: {str(e)}")


@router.post("/metrics/group/fairness/dir/request")
async def schedule_dir(request: GroupMetricRequest, background_tasks: BackgroundTasks):
    """Schedule a recurring computation of DIR metric."""
    request_id = str(uuid.uuid4())
    logger.info(f"Scheduling DIR computation with ID: {request_id}")
    # TODO: Implement background task scheduling
    return {"requestId": request_id}


@router.delete("/metrics/group/fairness/dir/request")
async def delete_dir_schedule(schedule: ScheduleId):
    """Delete a recurring computation of DIR metric."""
    logger.info(f"Deleting DIR schedule: {schedule.requestId}")
    # TODO: Implement schedule deletion
    return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}


@router.get("/metrics/group/fairness/dir/requests")
async def list_dir_requests():
    """List the currently scheduled computations of DIR metric."""
    # TODO: Implement request listing
    return {"requests": []}


# Deprecated DIR endpoints
@router.post("/dir", deprecated=True)
async def compute_dir_deprecated(request: GroupMetricRequest):
    """Compute the current value of Disparate Impact Ratio metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/dir instead.
    """
    return await compute_dir(request)


@router.get("/dir/definition", deprecated=True)
async def get_dir_definition_deprecated():
    """Provide a general definition of Disparate Impact Ratio metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/dir/definition instead.
    """
    return await get_dir_definition()


@router.post("/dir/definition", deprecated=True)
async def interpret_dir_value_deprecated(request: GroupDefinitionRequest):
    """Provide a specific interpretation of a DIR metric value (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/dir/definition instead.
    """
    return await interpret_dir_value(request)


@router.post("/dir/request", deprecated=True)
async def schedule_dir_deprecated(request: GroupMetricRequest, background_tasks: BackgroundTasks):
    """Schedule a recurring computation of DIR metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/dir/request instead.
    """
    return await schedule_dir(request, background_tasks)


@router.delete("/dir/request", deprecated=True)
async def delete_dir_schedule_deprecated(schedule: ScheduleId):
    """Delete a recurring computation of DIR metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/dir/request instead.
    """
    return await delete_dir_schedule(schedule)


@router.get("/dir/requests", deprecated=True)
async def list_dir_requests_deprecated():
    """List the currently scheduled computations of DIR metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/dir/requests instead.
    """
    return await list_dir_requests()
