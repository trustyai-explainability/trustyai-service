from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import pandas as pd
import uuid
import logging

from src.endpoints.metrics.fairness.group.utils import (
    GroupMetricRequest, GroupDefinitionRequest, ScheduleId,
    get_data_source, get_prometheus_scheduler, calculate_fairness_metric
    )
from src.service.prometheus.metric_value_carrier import MetricValueCarrier

router = APIRouter()
logger = logging.getLogger(__name__)


def calculate_dir_metric(
    request: GroupMetricRequest,
    dataframe: pd.DataFrame,
) -> MetricValueCarrier:
    """
    Calculate the Disparate Impact Ratio metric for the given dataframe and request.
    This function is registered with the metrics directory and called by the scheduler.
    """
    try:
        request.metric_name = "DIR"
        return calculate_fairness_metric(dataframe, request)
    except Exception as e:
        logger.error(f"Error calculating DIR: {str(e)}")
        raise e


# Register the DIR calculator with the metrics directory
def register_dir_calculator():
    """Register the DIR calculator with the global metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register("DIR", calculate_dir_metric)
        logger.info("DIR calculator registered with metrics directory")

# Register on module import
try:
    register_dir_calculator()
except Exception as e:
    logger.warning(f"Could not register DIR calculator on import: {e}")


# Disparate Impact Ratio
@router.post("/metrics/group/fairness/dir")
async def compute_dir(request: GroupMetricRequest, delta: Optional[float] = Query(None)):
    """Compute the current value of Disparate Impact Ratio metric."""
    try:
        logger.info(f"Computing DIR for model: {request.model_id}")
        request.metric_name = "DIR"

        # Get data source and load dataframe
        data_source = get_data_source()
        batch_size = request.batch_size if request.batch_size else 100

        # Get dataframe for the model
        dataframe = await data_source.get_organic_dataframe(request.model_id, batch_size)
        if dataframe.empty:
            raise HTTPException(status_code=404, detail=f"No data found for model: {request.model_id}")

        # Calculate DIR using our calculator
        result = calculate_dir_metric(dataframe, request)
        if delta is None:
            delta = 0.2
        return {
            "name": "DIR",
            "value": result.get_value(),
            "type": "FAIRNESS",
            "specificDefinition": f"Disparate Impact Ratio value of {result.get_value():.4f}",
            "thresholds": {
                "lowerBound": 1 - delta,
                "upperBound": 1 + delta,
                "outsideBounds": result.get_value() < 1 - delta or result.get_value() > 1 + delta
            }
        }
    except Exception as e:
        logger.error(f"Error computing DIR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}") from e

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
async def schedule_dir(request: GroupMetricRequest):
    """Schedule a recurring computation of DIR metric."""
    try:
        # Generate UUID for this request
        request_id = uuid.uuid4()
        logger.info(f"Scheduling DIR computation with ID: {request_id}")

        # Set metric name automatically
        request.metric_name = "DIR"

        # Get the scheduler and register the request
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Register with the scheduler (this will reconcile the request and store it)
        await scheduler.register("DIR", request_id, request)

        logger.info(f"Successfully scheduled DIR computation with ID: {request_id}")
        return {"requestId": str(request_id)}

    except Exception as e:
        logger.error(f"Error scheduling DIR computation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scheduling metric: {str(e)}") from e


@router.delete("/metrics/group/fairness/dir/request")
async def delete_dir_schedule(schedule: ScheduleId):
    """Delete a recurring computation of DIR metric."""
    try:
        logger.info(f"Deleting DIR schedule: {schedule.requestId}")

        # Get the scheduler and delete the request
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Convert string ID to UUID
        try:
            request_uuid = uuid.UUID(schedule.requestId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request ID format")

        # Delete from scheduler
        await scheduler.delete("DIR", request_uuid)

        logger.info(f"Successfully deleted DIR schedule: {schedule.requestId}")
        return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}

    except Exception as e:
        logger.error(f"Error deleting DIR schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting schedule: {str(e)}")


@router.get("/metrics/group/fairness/dir/requests")
async def list_dir_requests():
    """List the currently scheduled computations of DIR metric."""
    try:
        # Get the scheduler and list DIR requests
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Get all requests for DIR
        dir_requests = scheduler.get_requests("DIR")
        
        # Convert to list format expected by client
        requests_list = []
        for request_id, request in dir_requests.items():
            # Validate request object type before property access
            if hasattr(request, "model_id") and hasattr(request, "batch_size") and \
               hasattr(request, "protected_attribute") and hasattr(request, "outcome_name"):
                requests_list.append({
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": "DIR",
                    "batchSize": request.batch_size,
                    "protectedAttribute": request.protected_attribute,
                    "outcomeName": request.outcome_name
                })
            else:
                # Log warning for malformed request objects and skip them
                logger.warning(f"Skipping malformed DIR request {request_id}: missing required attributes")
                continue

        return {"requests": requests_list}
    except Exception as e:
        logger.error(f"Error listing DIR requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing requests: {str(e)}")


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
async def schedule_dir_deprecated(request: GroupMetricRequest):
    """Schedule a recurring computation of DIR metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/dir/request instead.
    """
    return await schedule_dir(request)


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
