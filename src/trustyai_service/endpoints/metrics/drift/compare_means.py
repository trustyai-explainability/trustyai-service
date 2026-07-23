"""Compare means endpoint for detecting drift through statistical comparison of means."""

import logging
import uuid
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trustyai_service.core.metrics.drift.compare_means import (
    DEFAULT_ALPHA,
    DEFAULT_EQUAL_VAR,
    DEFAULT_NAN_POLICY,
    CompareMeans,
    NanPolicy,
)
from trustyai_service.service.data.datasources.data_source import DataSource
from trustyai_service.service.data.shared_data_source import get_shared_data_source
from trustyai_service.service.payloads.metrics.base_metric_request import (
    BaseMetricRequest,
)
from trustyai_service.service.prometheus.prometheus_scheduler import PrometheusScheduler
from trustyai_service.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)
from trustyai_service.service.utils.logging_utils import log_deprecated_endpoint

router = APIRouter()
logger = logging.getLogger(__name__)

# Metric name constants
METRIC_NAME = "CompareMeans"
DEPRECATED_METRIC_NAME = "Meanshift"  # Legacy name for backwards compatibility

# Default parameter values
DEFAULT_BATCH_SIZE = 100
# Note: DEFAULT_ALPHA, DEFAULT_EQUAL_VAR, and DEFAULT_NAN_POLICY are imported
# from trustyai_service.core.metrics.drift.compare_means to ensure consistency


def get_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source() -> DataSource:
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    """Identifier for a scheduled metric computation request."""

    requestId: str


class CompareMeansMetricRequest(BaseMetricRequest):
    """Request parameters for compare means drift detection metric."""

    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: str | None = Field(
        default=None, alias="metricName"
    )  # Will be set by endpoint
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, alias="batchSize", gt=0)

    # CompareMeans-specific fields
    alpha: float = Field(default=DEFAULT_ALPHA, alias="alpha", gt=0, lt=1)
    equal_var: bool = Field(default=DEFAULT_EQUAL_VAR, alias="equalVar")
    nan_policy: NanPolicy = Field(default=DEFAULT_NAN_POLICY, alias="nanPolicy")
    reference_tag: str | None = Field(default=None, alias="referenceTag")
    fit_columns: list[str] = Field(default_factory=list, alias="fitColumns")

    @model_validator(mode="after")
    def _set_default_metric_name(self) -> "CompareMeansMetricRequest":
        """Automatically set metric_name to default if not provided."""
        if self.metric_name is None:
            self.metric_name = METRIC_NAME
        return self

    def retrieve_tags(self) -> dict[str, str]:
        """Retrieve tags for this CompareMeans metric request."""
        tags = self.retrieve_default_tags()
        if self.reference_tag:
            tags["referenceTag"] = self.reference_tag
        if self.fit_columns:
            tags["fitColumns"] = ",".join(self.fit_columns)
        return tags


@router.post("/metrics/drift/comparemeans")
async def compute_compare_means(
    request: CompareMeansMetricRequest,
) -> dict[str, float | bool | str | dict[str, dict[str, float | bool]]]:
    """Compute the current value of CompareMeans metric."""
    # Validate inputs before try block
    if not request.reference_tag or not request.reference_tag.strip():
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
    else:
        valid_features = [f.strip() for f in request.fit_columns if f.strip()]
        if not valid_features:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="fitColumns must contain at least one non-empty feature name",
            )
        request.fit_columns = valid_features

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

        # Validate data availability
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

        # Calculate t-test for each feature
        alpha = request.alpha
        equal_var = request.equal_var
        nan_policy = request.nan_policy

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

            results[feature_name] = CompareMeans.ttest_ind(
                reference_data=reference_data,
                current_data=current_data,
                alpha=alpha,
                equal_var=equal_var,
                nan_policy=nan_policy,
            )

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

    # Find the feature with the maximum absolute statistic
    # If drift is detected, prioritize features that detected drift to ensure
    # consistency: drift_detected=True implies p_value < alpha
    if drift_detected:
        # Among features that detected drift, find the one with max absolute statistic
        drifting_features = [r for r in results.values() if r["drift_detected"]]
        max_feature_result = max(drifting_features, key=lambda r: abs(r["statistic"]))
    else:
        # If no drift detected, use the feature with max absolute statistic overall
        max_feature_result = max(results.values(), key=lambda r: abs(r["statistic"]))

    max_statistic = max_feature_result["statistic"]
    corresponding_p_value = max_feature_result["p_value"]

    return {
        "status": "success",
        "value": abs(max_statistic),
        "drift_detected": drift_detected,
        "p_value": corresponding_p_value,
        "alpha": alpha,
        "feature_results": results,
    }


@router.get("/metrics/drift/comparemeans/definition")
async def get_compare_means_definition() -> dict[str, str]:
    """Provide a general definition of CompareMeans metric."""
    description = """The independent two-sample t-test is used to determine whether two independent samples
    have significantly different means. This implementation uses Welch's t-test by default (equal_var=False),
    which does not assume equal population variances.

    For more information, see the following:
    1. https://en.wikipedia.org/wiki/Student%27s_t-test
    2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    """

    return {
        "name": "Independent Two-Sample T-Test",
        "description": description,
    }


@router.post("/metrics/drift/comparemeans/request")
async def schedule_compare_means(request: CompareMeansMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of CompareMeans metric."""
    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    # Validate request before scheduling
    if not request.model_id or not request.model_id.strip():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="model_id is required and cannot be empty",
        )

    # Validate drift-specific required fields before scheduling
    if not request.reference_tag or not request.reference_tag.strip():
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
    else:
        valid_features = [f.strip() for f in request.fit_columns if f.strip()]
        if not valid_features:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="fitColumns must contain at least one non-empty feature name",
            )
        request.fit_columns = valid_features

    try:
        # Generate UUID for this request
        request_id = uuid.uuid4()
        logger.info("Scheduling %s computation with ID: %s.", METRIC_NAME, request_id)

        # Set metric name automatically
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
            "Successfully scheduled %s computation with ID: %s", METRIC_NAME, request_id
        )
        return {"requestId": str(request_id)}


@router.delete("/metrics/drift/comparemeans/request")
async def delete_compare_means_schedule(schedule: ScheduleId) -> dict[str, str]:
    """Delete a recurring computation of CompareMeans metric."""
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


@router.get("/metrics/drift/comparemeans/requests")
async def list_compare_means_requests() -> dict[str, list[dict[str, Any]]]:
    """List the currently scheduled computations of CompareMeans metric."""
    # Get the scheduler and validate availability
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    try:
        # Get all requests for CompareMeans
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
                        "alpha": getattr(request, "alpha", DEFAULT_ALPHA),
                        "equalVar": getattr(request, "equal_var", DEFAULT_EQUAL_VAR),
                        "nanPolicy": getattr(request, "nan_policy", DEFAULT_NAN_POLICY),
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
            detail="Error listing requests. Check server logs for details.",
        ) from e
    else:
        return {"requests": requests_list}


# ============================================================================
# DEPRECATED: Meanshift Endpoints (Backward Compatibility)
# ============================================================================
# These endpoints have been renamed to CompareMeans (METRIC_NAME).
# The "Meanshift" (DEPRECATED_METRIC_NAME) naming was deprecated as it may be confused
# with the Mean Shift clustering algorithm. These endpoints proxy to CompareMeans
# for backward compatibility and may be removed in a future version.
# ============================================================================


class MeanshiftMetricRequest(CompareMeansMetricRequest):
    """DEPRECATED: Use CompareMeansMetricRequest instead.

    Maintained for backward compatibility with old API clients.
    This class inherits from CompareMeansMetricRequest to ensure consistency
    and reduce duplication. All fields and validation behavior are inherited.
    """


@router.post("/metrics/drift/meanshift", deprecated=True)
async def compute_meanshift(
    request: MeanshiftMetricRequest,
) -> dict[str, float | bool | str | dict[str, dict[str, float | bool]]]:
    """Compute the current value of Meanshift metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/comparemeans
    instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    # Convert to CompareMeans request format
    # Use exclude_none=True so that omitted fields get their defaults applied
    compare_means_request = CompareMeansMetricRequest.model_validate(
        request.model_dump(exclude_none=True)
    )
    return await compute_compare_means(compare_means_request)


@router.get("/metrics/drift/meanshift/definition", deprecated=True)
async def get_meanshift_definition() -> dict[str, str]:
    """Provide a general definition of Meanshift metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/drift/comparemeans/definition instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await get_compare_means_definition()


@router.post("/metrics/drift/meanshift/request", deprecated=True)
async def schedule_meanshift(request: MeanshiftMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of Meanshift metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/drift/comparemeans/request instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    # Convert to CompareMeans request format
    # Use exclude_none=True so that omitted fields get their defaults applied
    compare_means_request = CompareMeansMetricRequest.model_validate(
        request.model_dump(exclude_none=True)
    )
    return await schedule_compare_means(compare_means_request)


@router.delete("/metrics/drift/meanshift/request", deprecated=True)
async def delete_meanshift_schedule(schedule: ScheduleId) -> dict[str, str]:
    """Delete a recurring computation of Meanshift metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/drift/comparemeans/request instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await delete_compare_means_schedule(schedule)


@router.get("/metrics/drift/meanshift/requests", deprecated=True)
async def list_meanshift_requests() -> dict[str, list[dict[str, Any]]]:
    """List the currently scheduled computations of Meanshift metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/drift/comparemeans/requests instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await list_compare_means_requests()
