from fastapi import APIRouter, HTTPException
import logging
import uuid
import pandas as pd

from src.service.prometheus.metric_value_carrier import MetricValueCarrier
from src.endpoints.metrics.fairness.group.utils import (
    GroupMetricRequest, GroupDefinitionRequest, ScheduleId,
    get_prometheus_scheduler, get_data_source, calculate_fairness_metric
    )

router = APIRouter()
logger = logging.getLogger(__name__)

# Note: SPDRequest class removed - using GroupMetricRequest for consistency with Java API

def calculate_spd_metric(dataframe: pd.DataFrame, request: GroupMetricRequest) -> MetricValueCarrier:
    """
    Calculate SPD metric for the given dataframe and request.
    This function is registered with the metrics directory and called by the scheduler.
    """
    try:
        return calculate_fairness_metric(dataframe, request)

    except Exception as e:
        logger.error(f"Error calculating SPD: {str(e)}")
        raise e

# Register the SPD calculator with the metrics directory
def register_spd_calculator():
    """Register the SPD calculator with the global metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register("SPD", calculate_spd_metric)
        logger.info("SPD calculator registered with metrics directory")

# Register on module import
try:
    register_spd_calculator()
except Exception as e:
    logger.warning(f"Could not register SPD calculator on import: {e}")


# Statistical Parity Difference
@router.post("/metrics/group/fairness/spd")
async def compute_spd(request: GroupMetricRequest):
    """Compute the current value of Statistical Parity Difference metric."""
    try:
        logger.info(f"Computing SPD for model: {request.model_id}")
        request.metric_name = "SPD"

        # Get data source and load dataframe
        data_source = get_data_source()
        batch_size = request.batch_size if request.batch_size else 100

        # Get dataframe for the model
        dataframe = await data_source.get_organic_dataframe(request.model_id, batch_size)

        if dataframe.empty:
            raise HTTPException(status_code=404, detail=f"No data found for model: {request.model_id}")

        # Calculate SPD using our calculator
        result = calculate_spd_metric(dataframe, request)

        return {
            "name": "SPD",
            "value": result.get_value(),
            "type": "FAIRNESS",
            "specificDefinition": f"Statistical Parity Difference value of {result.get_value():.4f}",
            "thresholds": {
                "lowerBound": -0.1,
                "upperBound": 0.1,
                "outsideBounds": abs(result.get_value()) > 0.1
            }
        }
    except Exception as e:
        logger.error(f"Error computing SPD: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error computing metric: {str(e)}"
        ) from e


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
        raise HTTPException(status_code=500, detail=f"Error interpreting value: {str(e)}")


@router.post("/metrics/group/fairness/spd/request")
async def schedule_spd(request: GroupMetricRequest):
    """Schedule a recurring computation of SPD metric."""
    try:
        # Generate UUID for this request
        request_id = uuid.uuid4()
        logger.info(f"Scheduling SPD computation with ID: {request_id}")

        # Set metric name automatically
        request.metric_name = "SPD"

        # Get the scheduler and register the request
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Register with the scheduler (this will reconcile the request and store it)
        await scheduler.register("SPD", request_id, request)

        logger.info(f"Successfully scheduled SPD computation with ID: {request_id}")
        return {"requestId": str(request_id)}

    except Exception as e:
        logger.error(f"Error scheduling SPD computation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error scheduling metric: {str(e)}"
        ) from e


@router.delete("/metrics/group/fairness/spd/request")
async def delete_spd_schedule(schedule: ScheduleId):
    """Delete a recurring computation of SPD metric."""
    try:
        logger.info(f"Deleting SPD schedule: {schedule.requestId}")

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
        await scheduler.delete("SPD", request_uuid)

        logger.info(f"Successfully deleted SPD schedule: {schedule.requestId}")
        return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}

    except Exception as e:
        logger.error(f"Error deleting SPD schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting schedule: {str(e)}")


@router.get("/metrics/group/fairness/spd/requests")
async def list_spd_requests():
    """List the currently scheduled computations of SPD metric."""
    try:
        # Get the scheduler and list SPD requests
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Get all requests for SPD
        spd_requests = scheduler.get_requests("SPD")

        # Convert to list format expected by client
        requests_list = []
        for request_id, request in spd_requests.items():
            # Validate request object type before property access
            if hasattr(request, "model_id") and hasattr(request, "batch_size") and \
               hasattr(request, "protected_attribute") and hasattr(request, "outcome_name"):
                requests_list.append({
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": "SPD",
                    "batchSize": request.batch_size,
                    "protectedAttribute": request.protected_attribute,
                    "outcomeName": request.outcome_name
                })
            else:
                # Log warning for malformed request objects and skip them
                logger.warning(f"Skipping malformed SPD request {request_id}: missing required attributes")
                continue

        return {"requests": requests_list}

    except Exception as e:
        logger.error(f"Error listing SPD requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing requests: {str(e)}")


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
async def schedule_spd_deprecated(request: GroupMetricRequest):
    """Schedule a recurring computation of SPD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/request instead.
    """
    return await schedule_spd(request)


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
