"""Disparate impact ratio (DIR) fairness metric endpoint."""

import logging
import uuid
from http import HTTPStatus
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from trustyai_service.core.metrics.fairness.group.disparate_impact_ratio import (
    DisparateImpactRatio,
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
DEFAULT_DIR_THRESHOLD_DELTA = 0.2  # Default threshold delta for DIR fairness bounds
DEFAULT_BATCH_SIZE = 100  # Default batch size for data retrieval
DIR_FAIRNESS_TARGET = 1  # Perfect fairness target for DIR (ratio of 1.0)


def calculate_dir_metric(
    dataframe: pd.DataFrame,
    request: GroupMetricRequest,
) -> MetricValueCarrier:
    """Calculate the Disparate Impact Ratio metric for the given dataframe and request.

    This function is registered with the metrics directory and called by the scheduler.

    Args:
        dataframe: Input data containing protected attributes and outcomes
        request: Metric request specifying groups and favorable outcomes

    Returns:
        MetricValueCarrier containing the DIR value

    """
    try:
        # Prepare data using utility function
        privileged_data, unprivileged_data, outcome_name, favorable_values = (
            prepare_fairness_data(dataframe, request)
        )

        # Check for sufficient data
        if len(privileged_data) == 0 or len(unprivileged_data) == 0:
            logger.warning(
                "Insufficient data for DIR calculation: privileged=%s, unprivileged=%s samples. Returning NaN.",
                len(privileged_data),
                len(unprivileged_data),
            )
            return MetricValueCarrier(float("nan"))

    except (
        Exception
    ):  # Broad catch intentional: log context before re-raising calculation error
        logger.exception("Error calculating DIR")
        raise

    # Validate favorable outcomes (moved outside try block to avoid TRY301)
    if len(favorable_values) == 0:
        msg = "No favorable outcomes specified for DIR calculation"
        raise ValueError(msg)

    # Prepare data in the format expected by DisparateImpactRatio
    privileged_array = privileged_data[[outcome_name]].to_numpy()
    unprivileged_array = unprivileged_data[[outcome_name]].to_numpy()

    # Calculate DIR using the core implementation
    dir_value = DisparateImpactRatio.calculate(
        privileged=privileged_array,
        unprivileged=unprivileged_array,
        favorable_outputs=favorable_values,
    )

    logger.debug("DIR calculation result: %s", dir_value)
    return MetricValueCarrier(dir_value)


# Register the DIR calculator with the metrics directory
def register_dir_calculator() -> None:
    """Register the DIR calculator with the global metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register("DIR", calculate_dir_metric)
        logger.info("DIR calculator registered with metrics directory")


# Register on module import
try:
    register_dir_calculator()
except (
    Exception
) as e:  # Intentional: registration failure should not break module import
    logger.warning("Could not register DIR calculator on import: %s", e)


# Disparate Impact Ratio
@router.post("/metrics/group/fairness/dir")
async def compute_dir(
    request: GroupMetricRequest,
    delta: Annotated[float | None, Query()] = None,
) -> dict[str, str | float | dict]:
    """Compute the current value of Disparate Impact Ratio metric."""
    try:
        logger.info("Computing DIR for model: %s", request.model_id)
        request.metric_name = "DIR"

        # Get data source and load dataframe
        data_source = get_data_source()
        batch_size = request.batch_size or DEFAULT_BATCH_SIZE

        # Get dataframe for the model
        dataframe = await data_source.get_organic_dataframe(
            request.model_id,
            batch_size,
        )

        # Validate data availability
        if dataframe.empty:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"No data found for model: {request.model_id}",
            )

        # Calculate DIR using our calculator
        result = calculate_dir_metric(dataframe, request)

        # Use delta from query parameter, then request.threshold_delta, then default
        if delta is None:
            delta = (
                request.threshold_delta
                if request.threshold_delta is not None
                else DEFAULT_DIR_THRESHOLD_DELTA
            )

        # Validate delta is non-negative
        if delta < 0:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Threshold delta must be non-negative",
            )

        return {
            "name": "DIR",
            "value": result.get_value(),
            "type": "metric",
            "specificDefinition": f"Disparate Impact Ratio value of {result.get_value():.4f}",
            "thresholds": {
                "lowerBound": DIR_FAIRNESS_TARGET - delta,
                "upperBound": DIR_FAIRNESS_TARGET + delta,
                "outsideBounds": result.get_value() < DIR_FAIRNESS_TARGET - delta
                or result.get_value() > DIR_FAIRNESS_TARGET + delta,
            },
        }

    except HTTPException:
        raise
    except Exception as e:  # Broad catch intentional: endpoint catch-all for unknown computation errors
        logger.exception("Error computing DIR")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error computing metric. Check server logs for details.",
        ) from e


@router.get("/metrics/group/fairness/dir/definition")
async def get_dir_definition() -> dict[str, str]:
    """Provide a general definition of Disparate Impact Ratio metric."""
    return {
        "name": "Disparate Impact Ratio",
        "description": "Measures the ratio of favorable outcome rates between unprivileged and privileged groups. "
        "A value of 1.0 indicates perfect fairness, while values significantly different from 1.0 indicate potential bias.",
    }


@router.post("/metrics/group/fairness/dir/definition")
async def interpret_dir_value(request: GroupDefinitionRequest) -> dict[str, str]:
    """Provide a specific, plain-english interpretation of a specific value of DIR metric."""
    logger.info("Interpreting DIR value for model: %s", request.modelId)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="DIR value interpretation is not yet implemented",
    )


@router.post("/metrics/group/fairness/dir/request")
async def schedule_dir(request: GroupMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of DIR metric."""
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
        logger.info("Scheduling DIR computation with ID: %s", request_id)

        # Set metric name automatically
        request.metric_name = "DIR"

        # Register with the scheduler (this will reconcile the request and store it)
        await scheduler.register("DIR", request_id, request)

    except Exception as e:  # Broad catch intentional: scheduler registration errors should not crash endpoint
        logger.exception("Error scheduling DIR computation")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error scheduling metric: {e!s}",
        ) from e
    else:
        logger.info("Successfully scheduled DIR computation with ID: %s", request_id)
        return {"requestId": str(request_id)}


@router.delete("/metrics/group/fairness/dir/request")
async def delete_dir_schedule(schedule: ScheduleId) -> dict[str, str]:
    """Delete a recurring computation of DIR metric."""
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
        logger.info("Deleting DIR schedule: %s", schedule.requestId)

        # Delete from scheduler
        await scheduler.delete("DIR", request_uuid)

    except HTTPException:
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown deletion errors
        logger.exception("Error deleting DIR schedule")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error deleting schedule: {e!s}",
        ) from e
    else:
        logger.info("Successfully deleted DIR schedule: %s", schedule.requestId)
        return {
            "status": "success",
            "message": f"Schedule {schedule.requestId} deleted",
        }


@router.get("/metrics/group/fairness/dir/requests")
async def list_dir_requests() -> dict[str, list[dict]]:
    """List the currently scheduled computations of DIR metric."""
    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Prometheus scheduler not available",
        )

    try:
        # Get all requests for DIR
        dir_requests = scheduler.get_requests("DIR")

        # Convert to list format expected by client
        requests_list = []
        for request_id, request in dir_requests.items():
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
                        "metricName": "DIR",
                        "batchSize": request.batch_size,
                        "protectedAttribute": request.protected_attribute,
                        "outcomeName": request.outcome_name,
                    },
                )
            else:
                # Log warning for malformed request objects and skip them
                logger.warning(
                    "Skipping malformed DIR request %s: missing required attributes",
                    request_id,
                )
                continue

    except HTTPException:
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown listing errors
        logger.exception("Error listing DIR requests")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error listing requests: {e!s}",
        ) from e
    else:
        return {"requests": requests_list}


# Deprecated DIR endpoints
@router.post("/dir", deprecated=True)
async def compute_dir_deprecated(
    request: GroupMetricRequest,
    delta: Annotated[float | None, Query()] = None,
) -> dict[str, str | float | dict]:
    """Compute the current value of Disparate Impact Ratio metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/dir
    instead.
    """
    return await compute_dir(request, delta)


@router.get("/dir/definition", deprecated=True)
async def get_dir_definition_deprecated() -> dict[str, str]:
    """Provide a general definition of Disparate Impact Ratio metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/dir/definition instead.
    """
    return await get_dir_definition()


@router.post("/dir/definition", deprecated=True)
async def interpret_dir_value_deprecated(
    request: GroupDefinitionRequest,
) -> dict[str, str]:
    """Provide a specific interpretation of a DIR metric value (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/dir/definition instead.
    """
    return await interpret_dir_value(request)


@router.post("/dir/request", deprecated=True)
async def schedule_dir_deprecated(request: GroupMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of DIR metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/dir/request instead.
    """
    return await schedule_dir(request)


@router.delete("/dir/request", deprecated=True)
async def delete_dir_schedule_deprecated(schedule: ScheduleId) -> dict[str, str]:
    """Delete a recurring computation of DIR metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/dir/request instead.
    """
    return await delete_dir_schedule(schedule)


@router.get("/dir/requests", deprecated=True)
async def list_dir_requests_deprecated() -> dict[str, list[dict]]:
    """List the currently scheduled computations of DIR metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/group/fairness/dir/requests instead.
    """
    return await list_dir_requests()
