import logging
import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.core.metrics.fairness.group.group_statistical_parity_difference import GroupStatisticalParityDifference
from src.endpoints.metrics.fairness.group.utils import (
    GroupDefinitionRequest,
    GroupMetricRequest,
    ScheduleId,
    get_data_source,
    get_prometheus_scheduler,
    prepare_fairness_data,
)
from src.service.prometheus.metric_value_carrier import MetricValueCarrier

router = APIRouter()
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPD_THRESHOLD_DELTA = 0.1  # Default threshold delta for SPD fairness bounds
DEFAULT_BATCH_SIZE = 100  # Default batch size for data retrieval
SPD_FAIRNESS_TARGET = 0  # Perfect fairness target for SPD (difference of 0.0)

# Note: SPDRequest class removed - using GroupMetricRequest for consistency with Java API


def calculate_spd_metric(dataframe: pd.DataFrame, request: GroupMetricRequest) -> MetricValueCarrier:
    """
    Calculate the Statistical Parity Difference metric for the given dataframe and request.

    This function is registered with the metrics directory and called by the scheduler.

    Args:
        dataframe: Input data containing protected attributes and outcomes
        request: Metric request specifying groups and favorable outcomes

    Returns:
        MetricValueCarrier containing the SPD value
    """
    try:
        # Prepare data using utility function
        privileged_data, unprivileged_data, outcome_name, favorable_values = prepare_fairness_data(dataframe, request)

        # Check for sufficient data
        if len(privileged_data) == 0 or len(unprivileged_data) == 0:
            logger.warning(
                f"Insufficient data for SPD calculation: privileged={len(privileged_data)}, unprivileged={len(unprivileged_data)} samples. Returning NaN."
            )
            return MetricValueCarrier(float("nan"))

        # Validate favorable outcomes
        if len(favorable_values) == 0:
            raise ValueError("No favorable outcomes specified for SPD calculation")

        # Prepare data in the format expected by GroupStatisticalParityDifference
        privileged_array = privileged_data[[outcome_name]].to_numpy()
        unprivileged_array = unprivileged_data[[outcome_name]].to_numpy()

        # Calculate SPD using the core implementation
        spd_value = GroupStatisticalParityDifference.calculate(
            privileged=privileged_array, unprivileged=unprivileged_array, favorable_outputs=favorable_values
        )

        logger.debug(f"SPD calculation result: {spd_value:.4f}")
        return MetricValueCarrier(spd_value)

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
async def compute_spd(request: GroupMetricRequest, delta: float | None = Query(None)):
    """Compute the current value of Statistical Parity Difference metric."""
    try:
        logger.info(f"Computing SPD for model: {request.model_id}")
        request.metric_name = "SPD"

        # Get data source and load dataframe
        data_source = get_data_source()
        batch_size = request.batch_size if request.batch_size else DEFAULT_BATCH_SIZE

        # Get dataframe for the model
        dataframe = await data_source.get_organic_dataframe(request.model_id, batch_size)

        if dataframe.empty:
            raise HTTPException(status_code=404, detail=f"No data found for model: {request.model_id}")

        # Calculate SPD using our calculator
        result = calculate_spd_metric(dataframe, request)

        # Use delta from query parameter, then request.threshold_delta, then default
        if delta is None:
            delta = request.threshold_delta if request.threshold_delta is not None else DEFAULT_SPD_THRESHOLD_DELTA

        return {
            "name": "SPD",
            "value": result.get_value(),
            "type": "FAIRNESS",
            "specificDefinition": f"Statistical Parity Difference value of {result.get_value():.4f}",
            "thresholds": {
                "lowerBound": SPD_FAIRNESS_TARGET - delta,
                "upperBound": SPD_FAIRNESS_TARGET + delta,
                "outsideBounds": abs(result.get_value() - SPD_FAIRNESS_TARGET) > delta,
            },
        }
    except Exception as e:
        logger.error(f"Error computing SPD: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}") from e


@router.get("/metrics/group/fairness/spd/definition")
async def get_spd_definition():
    """Provide a general definition of Statistical Parity Difference metric."""
    return {
        "name": "Statistical Parity Difference",
        "description": "Measures the difference in favorable outcome rates between unprivileged and privileged groups. "
        "A value of zero indicates perfect fairness. "
        "Positive or negative values indicate bias favoring one group over another.",
    }


@router.post("/metrics/group/fairness/spd/definition")
async def interpret_spd_value(request: GroupDefinitionRequest):
    """Provide a specific, plain-english interpretation of a specific value of SPD metric."""
    try:
        logger.info(f"Interpreting SPD value for model: {request.modelId}")
        # TODO: Implement interpretation
        return {"interpretation": "The SPD value indicates the difference in favorable outcome rates between groups."}
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
        raise HTTPException(status_code=500, detail=f"Error scheduling metric: {str(e)}") from e


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
            if (
                hasattr(request, "model_id")
                and hasattr(request, "batch_size")
                and hasattr(request, "protected_attribute")
                and hasattr(request, "outcome_name")
            ):
                requests_list.append({
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": "SPD",
                    "batchSize": request.batch_size,
                    "protectedAttribute": request.protected_attribute,
                    "outcomeName": request.outcome_name,
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
async def compute_spd_deprecated(request: GroupMetricRequest, delta: float | None = Query(None)):
    """Compute the current value of Statistical Parity Difference metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd instead.
    """
    return await compute_spd(request, delta)


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
