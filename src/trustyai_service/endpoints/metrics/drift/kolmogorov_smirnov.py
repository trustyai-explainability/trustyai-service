"""Kolmogorov-Smirnov test endpoint for drift detection."""

import logging
import uuid
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from trustyai_service.core.metrics.drift.kolmogorov_smirnov import KolmogorovSmirnov
from trustyai_service.service.data.datasources.data_source import DataSource
from trustyai_service.service.data.shared_data_source import get_shared_data_source
from trustyai_service.service.payloads.metrics.base_metric_request import (
    BaseMetricRequest,
)
from trustyai_service.service.prometheus.prometheus_scheduler import PrometheusScheduler
from trustyai_service.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Metric name constant
METRIC_NAME = "KSTest"


def get_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source() -> DataSource:
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    """Identifier for a scheduled metric computation request."""

    requestId: str


class KSTestMetricRequest(BaseMetricRequest):
    """Request parameters for Kolmogorov-Smirnov test drift detection metric."""

    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: str | None = Field(
        default=None, alias="metricName"
    )  # Will be set by endpoint
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int = Field(default=100, alias="batchSize")

    # KSTest-specific fields
    threshold_delta: float = Field(
        default=0.05, alias="thresholdDelta"
    )  # Default alpha value
    reference_tag: str | None = Field(default=None, alias="referenceTag")
    fit_columns: list[str] = Field(default_factory=list, alias="fitColumns")

    def retrieve_tags(self) -> dict[str, str]:
        """Retrieve tags for this KSTest metric request."""
        tags = self.retrieve_default_tags()
        if self.reference_tag:
            tags["referenceTag"] = self.reference_tag
        if self.fit_columns:
            tags["fitColumns"] = ",".join(self.fit_columns)
        return tags


@router.post("/metrics/drift/kstest")
async def compute_kstest(
    request: KSTestMetricRequest,
) -> dict[str, float | bool | str | dict[str, dict[str, float]]]:
    """Compute the current value of KSTest metric."""
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

    except HTTPException:
        raise
    except Exception as e:  # Broad catch intentional: endpoint catch-all for unknown computation errors
        logger.exception("Error computing %s", METRIC_NAME)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error computing metric: {e!s}",
        ) from e

    # Validate data availability (after try block to avoid TRY301)
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

    # Calculate KS test for each feature
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

        results[feature_name] = KolmogorovSmirnov.kstest(
            reference_data=reference_data,
            current_data=current_data,
            alpha=alpha,
        )

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
        "feature_results": results,
    }


@router.get("/metrics/drift/kstest/definition")
async def get_kstest_definition() -> dict[str, str]:
    """Provide a general definition of KSTest metric."""
    description = """The two-sampled Kolmogorov-Smirnov test is a nonparametric statistical test.
    It can be used to determine whether two underlying one-dimensional probability distributions differ.

    For more information, see the following:
    1. https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    """

    return {
        "name": "Kolmogorov-Smirnov Test",
        "description": description,
    }


@router.post("/metrics/drift/kstest/request")
async def schedule_kstest(request: KSTestMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of KSTest metric."""
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

        # Set metric name automatically
        request.metric_name = METRIC_NAME

        # Register with the scheduler (this will reconcile the request and store it)
        await scheduler.register(request.metric_name, request_id, request)

    except Exception as e:  # Broad catch intentional: scheduler registration errors should not crash endpoint
        logger.exception("Error scheduling %s computation", METRIC_NAME)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error scheduling metric: {e!s}",
        ) from e
    else:
        logger.info(
            "Successfully scheduled %s computation with ID: %s", METRIC_NAME, request_id
        )
        return {"requestId": str(request_id)}


@router.delete("/metrics/drift/kstest/request")
async def delete_kstest_schedule(schedule: ScheduleId) -> dict[str, str]:
    """Delete a recurring computation of KSTest metric."""
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
        await scheduler.delete(METRIC_NAME, request_uuid)

    except HTTPException:
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown deletion errors
        logger.exception("Error deleting %s schedule", METRIC_NAME)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error deleting schedule: {e!s}",
        ) from e
    else:
        logger.info(
            "Successfully deleted %s schedule: %s", METRIC_NAME, schedule.requestId
        )
        return {
            "status": "success",
            "message": f"Schedule {schedule.requestId} deleted",
        }


@router.get("/metrics/drift/kstest/requests")
async def list_kstest_requests() -> dict[str, list[dict[str, Any]]]:
    """List the currently scheduled computations of KSTest metric."""
    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    try:
        # Get all requests for KSTest
        requests = scheduler.get_requests(METRIC_NAME)

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
                requests_list.append(
                    {
                        "requestId": str(request_id),
                        "modelId": request.model_id,
                        "metricName": METRIC_NAME,
                        "batchSize": request.batch_size,
                        "referenceTag": request.reference_tag,
                        "fitColumns": request.fit_columns,
                    }
                )
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
            detail=f"Error listing requests: {e!s}",
        ) from e
    else:
        return {"requests": requests_list}
