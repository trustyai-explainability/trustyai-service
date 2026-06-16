"""Streaming KS test endpoint for detecting drift via approximate Kolmogorov–Smirnov test."""

import logging
import uuid
from http import HTTPStatus
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.core.metrics.drift.greenwald_khanna_quantile_sketch import EPSILON_DEFAULT
from src.core.metrics.drift.kolmogorov_smirnov_streaming import (
    KolmogorovSmirnovStreaming,
)
from src.service.data.datasources.data_source import DataSource
from src.service.data.shared_data_source import get_shared_data_source
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.metric_value_carrier import MetricValueCarrier
from src.service.prometheus.prometheus_scheduler import PrometheusScheduler
from src.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)
from src.service.utils.logging_utils import log_deprecated_endpoint

router = APIRouter()
logger = logging.getLogger(__name__)

# Metric name constants
METRIC_NAME = "KSTESTSTREAMING"
DEPRECATED_METRIC_NAME = "APPROXKSTEST"


def get_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source() -> DataSource:
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    """Identifier for a scheduled metric computation request."""

    requestId: str


class ApproxKSTestMetricRequest(BaseMetricRequest):
    """Request parameters for streaming KS test drift detection metric."""

    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: str = Field(default=METRIC_NAME, alias="metricName")
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int = Field(default=100, alias="batchSize")

    # ApproxKSTest-specific fields
    threshold_delta: float = Field(default=0.05, alias="thresholdDelta")
    reference_tag: str | None = Field(default=None, alias="referenceTag")
    fit_columns: list[str] = Field(default_factory=list, alias="fitColumns")

    # Streaming-specific field: epsilon for GK sketch accuracy
    # Must be in the interval (0, 0.5] to satisfy GK sketch requirements.
    # Values close to 0.5 cause algorithm degeneration (constant compression).
    epsilon: float = Field(
        default=EPSILON_DEFAULT,
        gt=0.0,
        le=0.5,
        description="Error parameter for GK sketch; must be in (0, 0.5]. Default: 0.01",
    )

    def retrieve_tags(self) -> dict[str, str]:
        """Retrieve tags for this ApproxKSTest metric request."""
        tags = self.retrieve_default_tags()
        if self.reference_tag:
            tags["referenceTag"] = self.reference_tag
        if self.fit_columns:
            tags["fitColumns"] = ",".join(self.fit_columns)
        # Always reflect explicitly configured epsilon, even if it's 0.0
        if self.epsilon is not None:
            tags["epsilon"] = str(self.epsilon)
        return tags


@router.post("/metrics/drift/ksteststreaming")
async def compute_ksteststreaming(
    request: ApproxKSTestMetricRequest,
) -> dict[str, float | bool | str | dict[str, dict[str, float]]]:
    """Compute the current value of KS Test Streaming metric."""
    # Validate inputs before try block
    if not request.reference_tag:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="referenceTag is required for drift detection",
        )

    if not request.fit_columns:
        data_source = get_data_source()
        metadata = await data_source.get_metadata(request.model_id)
        request.fit_columns = list(metadata.input_schema.items.keys())
        logger.info(
            "fitColumns not specified, using all input columns for model %s: %s",
            request.model_id,
            request.fit_columns,
        )

    try:
        logger.info("Computing %s for model: %s", METRIC_NAME, request.model_id)

        # Get data source
        data_source = get_data_source()
        batch_size = request.batch_size

        # Get reference dataframe (tagged with referenceTag)
        reference_df = await data_source.get_dataframe_by_tag(
            request.model_id, request.reference_tag
        )

        # Get current dataframe (most recent organic data)
        current_df = await data_source.get_organic_dataframe(
            request.model_id, batch_size
        )

        if len(reference_df) == 0:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"No reference data found for model: {request.model_id} with tag: {request.reference_tag}",
            )

        if len(current_df) == 0:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"No current data found for model: {request.model_id}",
            )

        # Get epsilon parameter
        epsilon = request.epsilon

        # Calculate approximate KS test for each feature
        alpha = request.threshold_delta

        # Multi-feature case: iterate over features
        results = {}
        for feature_name in request.fit_columns:
            if (
                feature_name not in reference_df.columns
                or feature_name not in current_df.columns
            ):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Feature {feature_name} not found in data",
                )

            reference_data = reference_df[feature_name].to_numpy()
            current_data = current_df[feature_name].to_numpy()

            # Use streaming KS test
            ks = KolmogorovSmirnovStreaming(epsilon=epsilon)
            ks.insert_reference_batch(reference_data)
            ks.insert_current_batch(current_data)

            results[feature_name] = ks.kstest(alpha=alpha)

    except HTTPException:
        raise
    except Exception as e:  # Broad catch intentional: endpoint catch-all for unknown computation errors
        logger.exception("Error computing %s", METRIC_NAME)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error computing metric. Check server logs for details.",
        ) from e

    # Aggregate: drift detected if any feature shows drift
    drift_detected = any(r["drift_detected"] for r in results.values())
    max_statistic = max(r["statistic"] for r in results.values())
    min_p_value = min(r["p_value"] for r in results.values())

    return {
        "status": "success",
        "value": max_statistic,
        "drift_detected": drift_detected,
        "p_value": min_p_value,
        "alpha": alpha,
        "epsilon": epsilon,
        "feature_results": results,
    }


@router.get("/metrics/drift/ksteststreaming/definition")
async def get_ksteststreaming_definition() -> dict[str, str]:
    """Provide a general definition of KS Test Streaming metric."""
    description = """The streaming two-sample Kolmogorov–Smirnov test is a space-efficient
    approximation of the classic KS test for detecting distribution drift.

    It uses Greenwald–Khanna quantile sketches to maintain approximate CDFs for both
    reference and current distributions, enabling computation of an approximate KS
    statistic with bounded error using sublinear space.

    The algorithm is based on:
    Lall, A., 2015. Data streaming algorithms for the Kolmogorov–Smirnov test.
    IEEE International Conference on Big Data (Big Data), pp. 95-104.
    https://doi.org/10.1109/BigData.2015.7363746

    The epsilon parameter controls the accuracy-space tradeoff:
    - Smaller epsilon provides better accuracy but requires more space
    - The KS statistic approximation error is bounded by 4*epsilon
    """

    return {
        "name": "Kolmogorov-Smirnov Test Streaming",
        "description": description,
    }


@router.post("/metrics/drift/ksteststreaming/request")
async def schedule_ksteststreaming(
    request: ApproxKSTestMetricRequest,
) -> dict[str, str]:
    """Schedule a recurring computation of KS Test Streaming metric."""
    if not request.fit_columns:
        data_source = get_data_source()
        metadata = await data_source.get_metadata(request.model_id)
        request.fit_columns = list(metadata.input_schema.items.keys())
        logger.info(
            "fitColumns not specified, using all input columns for model %s: %s",
            request.model_id,
            request.fit_columns,
        )

    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    try:
        # Generate UUID for this request
        request_id = uuid.uuid4()
        logger.info("Scheduling %s computation with ID: %s.", METRIC_NAME, request_id)

        if not request.metric_name:
            request.metric_name = METRIC_NAME

        # Register with the scheduler (this will reconcile the request and store it)
        await scheduler.register(request.metric_name, request_id, request)

    except HTTPException:
        raise
    except Exception as e:  # Broad catch intentional: scheduler registration errors should not crash endpoint
        logger.exception("Error scheduling %s computation", METRIC_NAME)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error scheduling metric. Check server logs for details.",
        ) from e
    else:
        logger.info(
            "Successfully scheduled %s computation with ID: %s",
            METRIC_NAME,
            request_id,
        )
        return {"requestId": str(request_id)}


@router.delete("/metrics/drift/ksteststreaming/request")
async def delete_ksteststreaming_schedule(
    schedule: ScheduleId, metric_name: str = METRIC_NAME
) -> dict[str, str]:
    """Delete a recurring computation of KS Test Streaming metric."""
    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    # Convert string ID to UUID
    try:
        request_uuid = uuid.UUID(schedule.requestId)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Invalid request ID format"
        ) from e

    try:
        logger.info("Deleting %s schedule: %s", METRIC_NAME, schedule.requestId)

        # Delete from scheduler
        await scheduler.delete(metric_name, request_uuid)

    except HTTPException:
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown deletion errors
        logger.exception("Error deleting %s schedule", METRIC_NAME)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error deleting schedule. Check server logs for details.",
        ) from e
    else:
        logger.info(
            "Successfully deleted %s schedule: %s", METRIC_NAME, schedule.requestId
        )
        return {
            "status": "success",
            "message": f"Schedule {schedule.requestId} deleted",
        }


@router.get("/metrics/drift/ksteststreaming/requests")
async def list_ksteststreaming_requests(
    metric_name: str = METRIC_NAME,
) -> dict[str, list[dict[str, Any]]]:
    """List the currently scheduled computations of KS Test Streaming metric."""
    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    try:
        # Get all requests for KSTestStreaming
        requests = scheduler.get_requests(metric_name)

        # Convert to list format expected by client
        requests_list = []
        for request_id, request in requests.items():
            # Validate request object type before property access
            if (
                hasattr(request, "model_id")
                and hasattr(request, "batch_size")
                and hasattr(request, "reference_tag")
                and hasattr(request, "fit_columns")
            ):
                request_dict = {
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": METRIC_NAME,
                    "batchSize": request.batch_size,
                    "referenceTag": request.reference_tag,
                    "fitColumns": request.fit_columns,
                }
                # Include epsilon if available
                if hasattr(request, "epsilon"):
                    request_dict["epsilon"] = request.epsilon
                requests_list.append(request_dict)
            else:
                # Log warning for malformed request objects and skip them
                logger.warning(
                    "Skipping malformed %s request %s: missing required attributes",
                    METRIC_NAME,
                    request_id,
                )
                continue

    except HTTPException:
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown listing errors
        logger.exception("Error listing %s requests", METRIC_NAME)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error listing requests. Check server logs for details.",
        ) from e
    else:
        return {"requests": requests_list}


# ============================================================================
# DEPRECATED ENDPOINTS (ApproxKSTest -> KSTestStreaming)
# ============================================================================


@router.post("/metrics/drift/approxkstest", deprecated=True)
async def compute_approxkstest_deprecated(
    request: ApproxKSTestMetricRequest,
) -> dict[str, float | bool | str | dict[str, dict[str, float]]]:
    """Compute the current value of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await compute_ksteststreaming(request)


@router.get("/metrics/drift/approxkstest/definition", deprecated=True)
async def get_approxkstest_definition_deprecated() -> dict[str, str]:
    """Provide a general definition of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming/definition instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await get_ksteststreaming_definition()


@router.post("/metrics/drift/approxkstest/request", deprecated=True)
async def schedule_approxkstest_deprecated(
    request: ApproxKSTestMetricRequest,
) -> dict[str, str]:
    """Schedule a recurring computation of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming/request instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    request.metric_name = DEPRECATED_METRIC_NAME
    return await schedule_ksteststreaming(request)


@router.delete("/metrics/drift/approxkstest/request", deprecated=True)
async def delete_approxkstest_schedule_deprecated(
    schedule: ScheduleId,
) -> dict[str, str]:
    """Delete a recurring computation of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming/request instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await delete_ksteststreaming_schedule(
        schedule, metric_name=DEPRECATED_METRIC_NAME
    )


@router.get("/metrics/drift/approxkstest/requests", deprecated=True)
async def list_approxkstest_requests_deprecated() -> dict[str, list[dict[str, Any]]]:
    """List the currently scheduled computations of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming/requests instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await list_ksteststreaming_requests(metric_name=DEPRECATED_METRIC_NAME)


async def calculate_ksteststreaming_metric(
    batch: pd.DataFrame,
    request: BaseMetricRequest,
) -> MetricValueCarrier:
    """Calculate KSTestStreaming metric for the Prometheus scheduler."""
    data_source = get_data_source()
    reference_df = await data_source.get_dataframe_by_tag(
        request.model_id, request.reference_tag
    )
    fit_columns = request.fit_columns or list(batch.columns)
    alpha = getattr(request, "threshold_delta", 0.05)

    named_values = {}
    for feature_name in fit_columns:
        if feature_name in reference_df.columns and feature_name in batch.columns:
            result = KolmogorovSmirnovStreaming.approx_kstest(
                reference_data=reference_df[feature_name].to_numpy(),
                current_data=batch[feature_name].to_numpy(),
                alpha=alpha,
            )
            named_values[feature_name] = result["statistic"]
    return MetricValueCarrier(named_values or 0.0)


def _register_ksteststreaming_calculator() -> None:
    """Register the KSTestStreaming calculator with the metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register(
            METRIC_NAME, calculate_ksteststreaming_metric
        )
        logger.info("%s calculator registered with metrics directory", METRIC_NAME)


try:
    _register_ksteststreaming_calculator()
except (AttributeError, TypeError) as e:
    logger.warning("Could not register %s calculator on import: %s", METRIC_NAME, e)
