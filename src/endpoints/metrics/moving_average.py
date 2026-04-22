import logging
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.service.data.shared_data_source import get_shared_data_source
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.metric_value_carrier import MetricValueCarrier
from src.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)
from src.service.utils.logging_utils import log_deprecated_endpoint

router = APIRouter()
logger = logging.getLogger(__name__)

METRIC_NAME = "MovingAverage"
DEPRECATED_METRIC_NAME = "Identity"
DEFAULT_BATCH_SIZE = 100


def get_prometheus_scheduler():
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source():
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    requestId: str


class MovingAverageRequest(BaseMetricRequest):
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: str | None = Field(default=None, alias="metricName")
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, alias="batchSize")

    column_name: str = Field(alias="columnName")
    lower_threshold: float | None = Field(default=None, alias="lowerThreshold")
    upper_threshold: float | None = Field(default=None, alias="upperThreshold")

    @model_validator(mode="after")
    def _set_default_metric_name(self) -> "MovingAverageRequest":
        if self.metric_name is None:
            self.metric_name = METRIC_NAME
        return self

    def retrieve_tags(self) -> dict[str, str]:
        tags = self.retrieve_default_tags()
        tags["columnName"] = self.column_name
        return tags


def calculate_moving_average_metric(
    dataframe: pd.DataFrame,
    request: MovingAverageRequest,
) -> MetricValueCarrier:
    """Calculate the mean of a column's values over the last N data points.

    Registered with the metrics directory and called by the scheduler.
    """
    column_name = request.column_name

    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in data")

    values = dataframe[column_name].to_numpy()
    numeric_values = values[~np.isnan(values.astype(float))]

    if len(numeric_values) == 0:
        logger.warning(
            "No numeric values found in column '%s'. Returning NaN.",
            column_name,
        )
        return MetricValueCarrier(float("nan"))

    mean_value = float(np.mean(numeric_values))
    logger.debug("Moving average calculation result for '%s': %.4f", column_name, mean_value)
    return MetricValueCarrier(mean_value)


def register_moving_average_calculator() -> None:
    """Register the MovingAverage calculator with the global metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register(METRIC_NAME, calculate_moving_average_metric)
        logger.info("MovingAverage calculator registered with metrics directory")


try:
    register_moving_average_calculator()
except Exception as e:
    logger.warning("Could not register MovingAverage calculator on import: %s", e)


# ============================================================================
# MovingAverage endpoints
# ============================================================================


@router.post("/metrics/movingaverage")
async def compute_moving_average(request: MovingAverageRequest) -> dict:
    """Compute the mean of a column's last N values."""
    try:
        logger.info(
            "Computing MovingAverage for model: %s, column: %s",
            request.model_id,
            request.column_name,
        )

        data_source = get_data_source()
        batch_size = request.batch_size or DEFAULT_BATCH_SIZE

        dataframe = await data_source.get_organic_dataframe(request.model_id, batch_size)

        if dataframe.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for model: {request.model_id}",
            )

        result = calculate_moving_average_metric(dataframe, request)
        value = result.get_value()

        response: dict = {
            "name": METRIC_NAME,
            "value": value,
            "type": "IDENTITY",
            "specificDefinition": (
                f"The moving average of the last {batch_size} values of "
                f"column '{request.column_name}' in model {request.model_id} "
                f"is {value:.4f}."
            ),
        }

        if request.lower_threshold is not None or request.upper_threshold is not None:
            lower = request.lower_threshold if request.lower_threshold is not None else float("-inf")
            upper = request.upper_threshold if request.upper_threshold is not None else float("inf")
            response["thresholds"] = {
                "lowerBound": request.lower_threshold,
                "upperBound": request.upper_threshold,
                "outsideBounds": value < lower or value > upper,
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error computing MovingAverage: %s", e)
        raise HTTPException(status_code=500, detail=f"Error computing metric: {e}") from e


@router.get("/metrics/movingaverage/definition")
async def get_moving_average_definition() -> dict:
    """Provide a general definition of the Moving Average metric."""
    return {
        "name": "Moving Average",
        "description": (
            "Returns the arithmetic mean of a model's last N feature or output values "
            "for a specified column, where N is the batch size."
        ),
    }


@router.post("/metrics/movingaverage/definition")
async def interpret_moving_average_value(request: MovingAverageRequest) -> dict:
    """Provide a specific, plain-english interpretation of a Moving Average value."""
    try:
        logger.info(
            "Interpreting MovingAverage value for model: %s, column: %s",
            request.model_id,
            request.column_name,
        )
        return {
            "interpretation": (
                f"The moving average of column '{request.column_name}' "
                f"tracks the mean of the last {request.batch_size} values, "
                f"useful for monitoring trends and detecting shifts."
            ),
        }
    except Exception as e:
        logger.error("Error interpreting MovingAverage value: %s", e)
        raise HTTPException(status_code=500, detail=f"Error interpreting value: {e}")


@router.post("/metrics/movingaverage/request")
async def schedule_moving_average(request: MovingAverageRequest) -> dict:
    """Schedule a recurring computation of Moving Average metric."""
    try:
        request_id = uuid.uuid4()
        logger.info("Scheduling MovingAverage computation with ID: %s", request_id)

        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        await scheduler.register(METRIC_NAME, request_id, request)

        logger.info("Successfully scheduled MovingAverage computation with ID: %s", request_id)
        return {"requestId": str(request_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling MovingAverage computation: %s", e)
        raise HTTPException(status_code=500, detail=f"Error scheduling metric: {e}") from e


@router.delete("/metrics/movingaverage/request")
async def delete_moving_average_schedule(schedule: ScheduleId) -> dict:
    """Delete a recurring computation of Moving Average metric."""
    try:
        logger.info("Deleting MovingAverage schedule: %s", schedule.requestId)

        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        try:
            request_uuid = uuid.UUID(schedule.requestId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request ID format")

        await scheduler.delete(METRIC_NAME, request_uuid)

        logger.info("Successfully deleted MovingAverage schedule: %s", schedule.requestId)
        return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting MovingAverage schedule: %s", e)
        raise HTTPException(status_code=500, detail=f"Error deleting schedule: {e}")


@router.get("/metrics/movingaverage/requests")
async def list_moving_average_requests() -> dict:
    """List the currently scheduled computations of Moving Average metric."""
    try:
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        ma_requests = scheduler.get_requests(METRIC_NAME)

        requests_list = []
        for request_id, request in ma_requests.items():
            if hasattr(request, "model_id") and hasattr(request, "column_name"):
                requests_list.append({
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": METRIC_NAME,
                    "batchSize": getattr(request, "batch_size", DEFAULT_BATCH_SIZE),
                    "columnName": request.column_name,
                    "lowerThreshold": getattr(request, "lower_threshold", None),
                    "upperThreshold": getattr(request, "upper_threshold", None),
                })
            else:
                logger.warning(
                    "Skipping malformed MovingAverage request %s: missing required attributes",
                    request_id,
                )
                continue

        return {"requests": requests_list}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing MovingAverage requests: %s", e)
        raise HTTPException(status_code=500, detail=f"Error listing requests: {e}")


# ============================================================================
# Deprecated Identity endpoints (backward compatibility)
# ============================================================================


class IdentityMetricRequest(MovingAverageRequest):
    """Deprecated: Use MovingAverageRequest instead."""


@router.post("/metrics/identity", deprecated=True)
async def compute_identity_metric(request: IdentityMetricRequest) -> dict:
    """Compute identity metric (deprecated). Use /metrics/movingaverage instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    moving_avg_request = MovingAverageRequest.model_validate(request.model_dump(exclude_none=True))
    return await compute_moving_average(moving_avg_request)


@router.get("/metrics/identity/definition", deprecated=True)
async def get_identity_definition() -> dict:
    """Get identity metric definition (deprecated). Use /metrics/movingaverage/definition instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await get_moving_average_definition()


@router.post("/metrics/identity/definition", deprecated=True)
async def interpret_identity_value(request: IdentityMetricRequest) -> dict:
    """Interpret identity metric value (deprecated). Use /metrics/movingaverage/definition instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    moving_avg_request = MovingAverageRequest.model_validate(request.model_dump(exclude_none=True))
    return await interpret_moving_average_value(moving_avg_request)


@router.post("/metrics/identity/request", deprecated=True)
async def schedule_identity(request: IdentityMetricRequest) -> dict:
    """Schedule identity metric (deprecated). Use /metrics/movingaverage/request instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    moving_avg_request = MovingAverageRequest.model_validate(request.model_dump(exclude_none=True))
    return await schedule_moving_average(moving_avg_request)


@router.delete("/metrics/identity/request", deprecated=True)
async def delete_identity_schedule(schedule: ScheduleId) -> dict:
    """Delete identity metric schedule (deprecated). Use /metrics/movingaverage/request instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await delete_moving_average_schedule(schedule)


@router.get("/metrics/identity/requests", deprecated=True)
async def list_identity_requests() -> dict:
    """List identity metric schedules (deprecated). Use /metrics/movingaverage/requests instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await list_moving_average_requests()
