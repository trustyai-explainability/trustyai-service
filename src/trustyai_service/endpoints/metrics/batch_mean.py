"""Batch Mean metric endpoint.

Each invocation computes the arithmetic mean of a column's last N data
points (controlled by ``batchSize``). The Prometheus scheduler calls this
every scrape interval, exposing the result as a gauge. Because the data
window slides forward with each new inference, the time series of scraped
gauge values forms a moving average — Prometheus handles the temporal
dimension, so this code only needs to produce a single scalar per call.
"""

import logging
import uuid
from http import HTTPStatus

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trustyai_service.service.data.datasources.data_source import DataSource
from trustyai_service.service.data.shared_data_source import get_shared_data_source
from trustyai_service.service.payloads.metrics.base_metric_request import (
    BaseMetricRequest,
)
from trustyai_service.service.prometheus.metric_value_carrier import MetricValueCarrier
from trustyai_service.service.prometheus.prometheus_scheduler import PrometheusScheduler
from trustyai_service.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)
from trustyai_service.service.utils.logging_utils import log_deprecated_endpoint

router = APIRouter()
logger = logging.getLogger(__name__)

METRIC_NAME = "BatchMean"
DEPRECATED_METRIC_NAME = "Identity"
DEFAULT_BATCH_SIZE = 100


def get_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source() -> DataSource:
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    """Identifier for a scheduled metric computation."""

    requestId: str


class BatchMeanRequest(BaseMetricRequest):
    """Request parameters for the Batch Mean metric."""

    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: str | None = Field(default=None, alias="metricName")
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, alias="batchSize")

    column_name: str = Field(alias="columnName")
    lower_threshold: float | None = Field(default=None, alias="lowerThreshold")
    upper_threshold: float | None = Field(default=None, alias="upperThreshold")

    @model_validator(mode="after")
    def _set_default_metric_name(self) -> "BatchMeanRequest":
        if self.metric_name is None:
            self.metric_name = METRIC_NAME
        return self

    def retrieve_tags(self) -> dict[str, str]:
        """Return Prometheus labels for this metric request."""
        tags = self.retrieve_default_tags()
        tags["columnName"] = self.column_name
        return tags


def calculate_batch_mean_metric(
    dataframe: pd.DataFrame,
    request: BatchMeanRequest,
) -> MetricValueCarrier:
    """Calculate the mean of a column's values over the last N data points.

    Registered with the metrics directory and called by the scheduler.
    """
    column_name = request.column_name

    if column_name not in dataframe.columns:
        msg = f"Column '{column_name}' not found in data"
        raise ValueError(msg)

    values = dataframe[column_name].to_numpy()
    try:
        float_values = values.astype(float)
    except (ValueError, TypeError) as e:
        msg = f"Column '{column_name}' contains non-numeric data"
        raise TypeError(msg) from e
    numeric_values = float_values[~np.isnan(float_values)]

    if len(numeric_values) == 0:
        logger.warning(
            "No numeric values found in column '%s'. Returning NaN.",
            column_name,
        )
        return MetricValueCarrier(float("nan"))

    mean_value = float(np.mean(numeric_values))
    logger.debug(
        "Batch mean calculation result for '%s': %.4f", column_name, mean_value
    )
    return MetricValueCarrier(mean_value)


def register_batch_mean_calculator() -> None:
    """Register the BatchMean calculator with the global metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register(METRIC_NAME, calculate_batch_mean_metric)
        logger.info("BatchMean calculator registered with metrics directory")


try:
    register_batch_mean_calculator()
except (AttributeError, TypeError) as e:
    logger.warning("Could not register BatchMean calculator on import: %s", e)


# ============================================================================
# BatchMean endpoints
# ============================================================================


@router.post("/metrics/batchmean")
async def compute_batch_mean(request: BatchMeanRequest) -> dict:
    """Compute the mean of a column's last N values."""
    logger.info(
        "Computing BatchMean for model: %s, column: %s",
        request.model_id,
        request.column_name,
    )

    data_source = get_data_source()
    batch_size = request.batch_size or DEFAULT_BATCH_SIZE

    dataframe = await data_source.get_organic_dataframe(request.model_id, batch_size)

    if dataframe.empty:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No data found for model: {request.model_id}",
        )

    result = calculate_batch_mean_metric(dataframe, request)
    value = result.get_value()

    response: dict = {
        "name": METRIC_NAME,
        "value": value,
        "type": "BATCH_MEAN",
        "specificDefinition": (
            f"The batch mean of the last {batch_size} values of "
            f"column '{request.column_name}' in model {request.model_id} "
            f"is {value:.4f}."
        ),
    }

    if request.lower_threshold is not None or request.upper_threshold is not None:
        lower = (
            request.lower_threshold
            if request.lower_threshold is not None
            else float("-inf")
        )
        upper = (
            request.upper_threshold
            if request.upper_threshold is not None
            else float("inf")
        )
        response["thresholds"] = {
            "lowerBound": request.lower_threshold,
            "upperBound": request.upper_threshold,
            "outsideBounds": value < lower or value > upper,
        }

    return response


@router.get("/metrics/batchmean/definition")
async def get_batch_mean_definition() -> dict:
    """Provide a general definition of the Batch Mean metric."""
    return {
        "name": "Batch Mean",
        "description": (
            "Returns the arithmetic mean of a model's last N feature or output values "
            "for a specified column, where N is the batch size."
        ),
    }


@router.post("/metrics/batchmean/definition")
async def interpret_batch_mean_value(request: BatchMeanRequest) -> dict:
    """Provide a specific, plain-english interpretation of a Batch Mean value."""
    return {
        "interpretation": (
            f"The batch mean of column '{request.column_name}' "
            f"tracks the mean of the last {request.batch_size} values, "
            f"useful for monitoring trends and detecting shifts."
        ),
    }


@router.post("/metrics/batchmean/request")
async def schedule_batch_mean(request: BatchMeanRequest) -> dict:
    """Schedule a recurring computation of Batch Mean metric."""
    request_id = uuid.uuid4()
    logger.info("Scheduling BatchMean computation with ID: %s", request_id)

    scheduler = get_prometheus_scheduler()
    await scheduler.register(METRIC_NAME, request_id, request)

    logger.info("Successfully scheduled BatchMean computation with ID: %s", request_id)
    return {"requestId": str(request_id)}


@router.delete("/metrics/batchmean/request")
async def delete_batch_mean_schedule(schedule: ScheduleId) -> dict:
    """Delete a recurring computation of Batch Mean metric."""
    logger.info("Deleting BatchMean schedule: %s", schedule.requestId)

    scheduler = get_prometheus_scheduler()

    try:
        request_uuid = uuid.UUID(schedule.requestId)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Invalid request ID format"
        ) from e

    await scheduler.delete(METRIC_NAME, request_uuid)

    logger.info("Successfully deleted BatchMean schedule: %s", schedule.requestId)
    return {
        "status": "success",
        "message": f"Schedule {schedule.requestId} deleted",
    }


@router.get("/metrics/batchmean/requests")
async def list_batch_mean_requests() -> dict:
    """List the currently scheduled computations of Batch Mean metric."""
    scheduler = get_prometheus_scheduler()
    bm_requests = scheduler.get_requests(METRIC_NAME)

    requests_list = []
    for request_id, request in bm_requests.items():
        if hasattr(request, "model_id") and hasattr(request, "column_name"):
            requests_list.append(
                {
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": METRIC_NAME,
                    "batchSize": getattr(request, "batch_size", DEFAULT_BATCH_SIZE),
                    "columnName": request.column_name,
                    "lowerThreshold": getattr(request, "lower_threshold", None),
                    "upperThreshold": getattr(request, "upper_threshold", None),
                }
            )
        else:
            logger.warning(
                "Skipping malformed BatchMean request %s: missing required attributes",
                request_id,
            )

    return {"requests": requests_list}


# ============================================================================
# Deprecated Identity endpoints (backward compatibility)
# ============================================================================


class IdentityMetricRequest(BatchMeanRequest):
    """Deprecated: Use BatchMeanRequest instead."""


@router.post("/metrics/identity", deprecated=True)
async def compute_identity_metric(request: IdentityMetricRequest) -> dict:
    """Compute identity metric (deprecated). Use /metrics/batchmean instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await compute_batch_mean(request)


@router.get("/metrics/identity/definition", deprecated=True)
async def get_identity_definition() -> dict:
    """Get identity metric definition (deprecated). Use /metrics/batchmean/definition instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await get_batch_mean_definition()


@router.post("/metrics/identity/definition", deprecated=True)
async def interpret_identity_value(request: IdentityMetricRequest) -> dict:
    """Interpret identity metric value (deprecated). Use /metrics/batchmean/definition instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await interpret_batch_mean_value(request)


@router.post("/metrics/identity/request", deprecated=True)
async def schedule_identity(request: IdentityMetricRequest) -> dict:
    """Schedule identity metric (deprecated). Use /metrics/batchmean/request instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await schedule_batch_mean(request)


@router.delete("/metrics/identity/request", deprecated=True)
async def delete_identity_schedule(schedule: ScheduleId) -> dict:
    """Delete identity metric schedule (deprecated). Use /metrics/batchmean/request instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await delete_batch_mean_schedule(schedule)


@router.get("/metrics/identity/requests", deprecated=True)
async def list_identity_requests() -> dict:
    """List identity metric schedules (deprecated). Use /metrics/batchmean/requests instead."""
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await list_batch_mean_requests()
