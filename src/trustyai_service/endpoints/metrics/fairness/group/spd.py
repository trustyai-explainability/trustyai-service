"""Statistical parity difference (SPD) fairness metric endpoint."""

import logging
import uuid
from http import HTTPStatus
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from trustyai_service.core.metrics.fairness.group.group_statistical_parity_difference import (
    GroupStatisticalParityDifference,
)
from trustyai_service.endpoints.metrics.fairness.group.utils import (
    GroupDefinitionRequest,
    GroupMetricRequest,
    ScheduleId,
    get_data_source,
    get_prometheus_scheduler,
    prepare_fairness_data,
)
from trustyai_service.service.prometheus.metric_value_carrier import MetricValueCarrier

router = APIRouter()
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPD_THRESHOLD_DELTA = 0.1  # Default threshold delta for SPD fairness bounds
DEFAULT_BATCH_SIZE = 100  # Default batch size for data retrieval
SPD_FAIRNESS_TARGET = 0  # Perfect fairness target for SPD (difference of 0.0)

# Note: SPDRequest class removed - using GroupMetricRequest for consistency with Java API


def calculate_spd_metric(
    dataframe: pd.DataFrame,
    request: GroupMetricRequest,
) -> MetricValueCarrier:
    """Calculate the Statistical Parity Difference metric for the given dataframe and request.

    This function is registered with the metrics directory and called by the scheduler.

    Args:
        dataframe: Input data containing protected attributes and outcomes
        request: Metric request specifying groups and favorable outcomes

    Returns:
        MetricValueCarrier containing the SPD value

    """
    try:
        # Prepare data using utility function
        privileged_data, unprivileged_data, outcome_name, favorable_values = (
            prepare_fairness_data(dataframe, request)
        )

        # Check for sufficient data
        if len(privileged_data) == 0 or len(unprivileged_data) == 0:
            logger.warning(
                "Insufficient data for SPD calculation: privileged=%s, unprivileged=%s samples. Returning NaN.",
                len(privileged_data),
                len(unprivileged_data),
            )
            return MetricValueCarrier(float("nan"))

    except (
        Exception
    ):  # Broad catch intentional: log context before re-raising calculation error
        logger.exception("Error calculating SPD")
        raise

    # Validate favorable outcomes (moved outside try block to avoid TRY301)
    if len(favorable_values) == 0:
        msg = "No favorable outcomes specified for SPD calculation"
        raise ValueError(msg)

    # Prepare data in the format expected by GroupStatisticalParityDifference
    privileged_array = privileged_data[[outcome_name]].to_numpy()
    unprivileged_array = unprivileged_data[[outcome_name]].to_numpy()

    # Calculate SPD using the core implementation
    spd_value = GroupStatisticalParityDifference.calculate(
        privileged=privileged_array,
        unprivileged=unprivileged_array,
        favorable_outputs=favorable_values,
    )

    logger.debug("SPD calculation result: %s", spd_value)
    return MetricValueCarrier(spd_value)


# Register the SPD calculator with the metrics directory
def register_spd_calculator() -> None:
    """Register the SPD calculator with the global metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register("SPD", calculate_spd_metric)
        logger.info("SPD calculator registered with metrics directory")


# Register on module import
try:
    register_spd_calculator()
except (
    Exception
) as e:  # Intentional: registration failure should not break module import
    logger.warning("Could not register SPD calculator on import: %s", e)


# Statistical Parity Difference
@router.post("/metrics/group/fairness/spd")
async def compute_spd(
    request: GroupMetricRequest,
    delta: Annotated[float | None, Query()] = None,
) -> dict[str, str | float | dict]:
    """Compute the current value of Statistical Parity Difference metric."""
    try:
        logger.info("Computing SPD for model: %s", request.model_id)
        request.metric_name = "SPD"

        # Get data source and load dataframe
        data_source = get_data_source()
        batch_size = request.batch_size or DEFAULT_BATCH_SIZE

        # Get dataframe for the model
        dataframe = await data_source.get_organic_dataframe(
            request.model_id,
            batch_size,
        )

    except HTTPException:
        raise
    except Exception as e:  # Broad catch intentional: endpoint catch-all for unknown computation errors
        logger.exception("Error computing SPD")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error computing metric: {e!s}",
        ) from e

    # Validate data availability (moved outside try block to avoid TRY301)
    if dataframe.empty:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No data found for model: {request.model_id}",
        )

    # Calculate SPD using our calculator
    result = calculate_spd_metric(dataframe, request)

    # Use delta from query parameter, then request.threshold_delta, then default
    if delta is None:
        delta = (
            request.threshold_delta
            if request.threshold_delta is not None
            else DEFAULT_SPD_THRESHOLD_DELTA
        )

    return {
        "name": "SPD",
        "value": result.get_value(),
        "type": "metric",
        "specificDefinition": f"Statistical Parity Difference value of {result.get_value():.4f}",
        "thresholds": {
            "lowerBound": SPD_FAIRNESS_TARGET - delta,
            "upperBound": SPD_FAIRNESS_TARGET + delta,
            "outsideBounds": abs(result.get_value() - SPD_FAIRNESS_TARGET) > delta,
        },
    }


@router.get("/metrics/group/fairness/spd/definition")
async def get_spd_definition() -> dict[str, str]:
    """Provide a general definition of Statistical Parity Difference metric."""
    return {
        "name": "Statistical Parity Difference",
        "description": "Measures the difference in favorable outcome rates between unprivileged and privileged groups. "
        "A value of zero indicates perfect fairness. "
        "Positive or negative values indicate bias favoring one group over another.",
    }


@router.post("/metrics/group/fairness/spd/definition")
async def interpret_spd_value(request: GroupDefinitionRequest) -> dict[str, str]:
    """Provide a specific, plain-english interpretation of a specific value of SPD metric."""
    logger.info("Interpreting SPD value for model: %s", request.modelId)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="SPD value interpretation is not yet implemented",
    )


@router.post("/metrics/group/fairness/spd/request")
async def schedule_spd(request: GroupMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of SPD metric."""
    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Prometheus scheduler not available",
        )

    try:
        # Generate UUID for this request
        request_id = uuid.uuid4()
        logger.info("Scheduling SPD computation with ID: %s", request_id)

        # Set metric name automatically
        request.metric_name = "SPD"

        # Register with the scheduler (this will reconcile the request and store it)
        await scheduler.register("SPD", request_id, request)

    except Exception as e:  # Broad catch intentional: scheduler registration errors should not crash endpoint
        logger.exception("Error scheduling SPD computation")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error scheduling metric: {e!s}",
        ) from e
    else:
        logger.info("Successfully scheduled SPD computation with ID: %s", request_id)
        return {"requestId": str(request_id)}


@router.delete("/metrics/group/fairness/spd/request")
async def delete_spd_schedule(schedule: ScheduleId) -> dict[str, str]:
    """Delete a recurring computation of SPD metric."""
    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Prometheus scheduler not available",
        )

    # Convert string ID to UUID
    try:
        request_uuid = uuid.UUID(schedule.requestId)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Invalid request ID format",
        ) from e

    try:
        logger.info("Deleting SPD schedule: %s", schedule.requestId)

        # Delete from scheduler
        await scheduler.delete("SPD", request_uuid)

    except HTTPException:
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown deletion errors
        logger.exception("Error deleting SPD schedule")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error deleting schedule: {e!s}",
        ) from e
    else:
        logger.info("Successfully deleted SPD schedule: %s", schedule.requestId)
        return {
            "status": "success",
            "message": f"Schedule {schedule.requestId} deleted",
        }


@router.get("/metrics/group/fairness/spd/requests")
async def list_spd_requests() -> dict[str, list[dict]]:
    """List the currently scheduled computations of SPD metric."""
    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Prometheus scheduler not available",
        )

    try:
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
                requests_list.append(
                    {
                        "requestId": str(request_id),
                        "modelId": request.model_id,
                        "metricName": "SPD",
                        "batchSize": request.batch_size,
                        "protectedAttribute": request.protected_attribute,
                        "outcomeName": request.outcome_name,
                    },
                )
            else:
                # Log warning for malformed request objects and skip them
                logger.warning(
                    "Skipping malformed SPD request %s: missing required attributes",
                    request_id,
                )
                continue

    except HTTPException:
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown listing errors
        logger.exception("Error listing SPD requests")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error listing requests: {e!s}",
        ) from e
    else:
        return {"requests": requests_list}


# Deprecated SPD endpoints
@router.post("/spd", deprecated=True)
async def compute_spd_deprecated(
    request: GroupMetricRequest,
    delta: Annotated[float | None, Query()] = None,
) -> dict[str, str | float | dict]:
    """Compute the current value of Statistical Parity Difference metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd
    instead.
    """
    return await compute_spd(request, delta)


@router.get("/spd/definition", deprecated=True)
async def get_spd_definition_deprecated() -> dict[str, str]:
    """Provide a general definition of Statistical Parity Difference metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/spd/definition instead.
    """
    return await get_spd_definition()


@router.post("/spd/definition", deprecated=True)
async def interpret_spd_value_deprecated(
    request: GroupDefinitionRequest,
) -> dict[str, str]:
    """Provide a specific interpretation of a SPD metric value (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/spd/definition instead.
    """
    return await interpret_spd_value(request)


@router.post("/spd/request", deprecated=True)
async def schedule_spd_deprecated(request: GroupMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of SPD metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/spd/request instead.
    """
    return await schedule_spd(request)


@router.delete("/spd/request", deprecated=True)
async def delete_spd_schedule_deprecated(schedule: ScheduleId) -> dict[str, str]:
    """Delete a recurring computation of SPD metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/spd/request instead.
    """
    return await delete_spd_schedule(schedule)


@router.get("/spd/requests", deprecated=True)
async def list_spd_requests_deprecated() -> dict[str, list[dict]]:
    """List the currently scheduled computations of SPD metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/spd/requests instead.
    """
    return await list_spd_requests()
