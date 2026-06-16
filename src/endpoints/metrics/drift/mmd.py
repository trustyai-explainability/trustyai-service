"""MMD (Maximum Mean Discrepancy) endpoint for multivariate drift detection."""

import logging
import uuid
from http import HTTPStatus
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.core.metrics.drift.mmd import (
    DEFAULT_ALPHA,
    DEFAULT_BANDWIDTH,
    DEFAULT_KERNEL,
    DEFAULT_NUM_PERMUTATIONS,
    MMD,
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

METRIC_NAME = "MMD"
DEPRECATED_METRIC_NAME = "FOURIERMMD"
DEFAULT_BATCH_SIZE = 100


def get_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source() -> DataSource:
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    """Identifier for a scheduled metric computation request."""

    requestId: str


class MMDMetricRequest(BaseMetricRequest):
    """Request parameters for MMD drift detection metric."""

    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: str = Field(default=METRIC_NAME, alias="metricName")
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int | None = Field(default=DEFAULT_BATCH_SIZE, alias="batchSize", gt=0)

    reference_tag: str | None = Field(default=None, alias="referenceTag")
    fit_columns: list[str] = Field(default_factory=list, alias="fitColumns")

    # MMD-specific parameters
    num_permutations: int = Field(
        default=DEFAULT_NUM_PERMUTATIONS, alias="numPermutations", gt=0
    )
    bandwidth: float = Field(default=DEFAULT_BANDWIDTH, gt=0)
    kernel: str = Field(default=DEFAULT_KERNEL)
    alpha: float = Field(default=DEFAULT_ALPHA, gt=0, lt=1)
    seed: int | None = Field(default=None)

    def retrieve_tags(self) -> dict[str, str]:
        """Retrieve tags for this MMD metric request."""
        tags = self.retrieve_default_tags()
        if self.reference_tag:
            tags["referenceTag"] = self.reference_tag
        if self.fit_columns:
            tags["fitColumns"] = ",".join(self.fit_columns)
        return tags


async def _validate_drift_request(request: MMDMetricRequest) -> list[str]:
    """Validate common drift request fields and return cleaned fit_columns."""
    if not request.model_id or not request.model_id.strip():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="model_id is required and cannot be empty",
        )
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
        return request.fit_columns

    valid_features = [f.strip() for f in request.fit_columns if f.strip()]
    if not valid_features:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="fitColumns must contain at least one non-empty feature name",
        )
    return valid_features


@router.post("/metrics/drift/mmd")
async def compute_mmd(
    request: MMDMetricRequest,
) -> dict[str, float | bool | str | list[str]]:
    """Compute the current value of MMD metric."""
    valid_features = await _validate_drift_request(request)
    logger.info("Computing %s for model: %s", METRIC_NAME, request.model_id)

    data_source = get_data_source()

    try:
        reference_df = await data_source.get_dataframe_by_tag(
            request.model_id, request.reference_tag
        )
        current_df = await data_source.get_organic_dataframe(
            request.model_id, request.batch_size or DEFAULT_BATCH_SIZE
        )
    except (
        Exception
    ) as e:  # Broad catch intentional: data-fetch errors should not crash endpoint
        logger.exception("Error fetching data for %s", METRIC_NAME)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error fetching data. Check server logs for details.",
        ) from e

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

    available_cols = set(reference_df.columns) & set(current_df.columns)
    missing = [f for f in valid_features if f not in available_cols]
    if missing:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Feature {missing[0]} not found in data",
        )

    reference_data = reference_df[valid_features].to_numpy()
    current_data = current_df[valid_features].to_numpy()

    try:
        result = MMD.compute(
            reference_data=reference_data,
            current_data=current_data,
            alpha=request.alpha,
            seed=request.seed,
            num_permutations=request.num_permutations,
            bandwidth=request.bandwidth,
            kernel=request.kernel,
        )
    except ImportError:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="MMD requires the 'goodpoints' package. Install with: pip install trustyai-service[mmd]",
        ) from None
    except Exception as e:  # Broad catch intentional: endpoint catch-all for unknown computation errors
        logger.exception("Error computing %s", METRIC_NAME)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error computing metric. Check server logs for details.",
        ) from e

    return {
        "status": "success",
        "value": result["statistic"],
        "drift_detected": result["drift_detected"],
        "p_value": result["p_value"],
        "threshold": result["threshold"],
        "alpha": result["alpha"],
        "fit_columns": valid_features,
    }


@router.get("/metrics/drift/mmd/definition")
async def get_mmd_definition() -> dict[str, str]:
    """Provide a general definition of MMD metric."""
    return {
        "name": "Maximum Mean Discrepancy (MMD)",
        "description": (
            "A multivariate two-sample test using Maximum Mean Discrepancy (MMD). "
            "Supports Compress Then Test (CTT, default) for near-linear time with "
            "optimal power, and Random Fourier Features (RFF) for kernel approximation. "
            "Powered by Microsoft's goodpoints package. "
            "Reference: Domingo-Enrich, Dwivedi & Mackey "
            "(AISTATS 2023, arXiv:2301.05974)."
        ),
    }


@router.post("/metrics/drift/mmd/request")
async def schedule_mmd(request: MMDMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of MMD metric."""
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    await _validate_drift_request(request)

    try:
        request_id = uuid.uuid4()
        logger.info("Scheduling %s computation with ID: %s.", METRIC_NAME, request_id)
        if not request.metric_name:
            request.metric_name = METRIC_NAME
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


@router.delete("/metrics/drift/mmd/request")
async def delete_mmd_schedule(
    schedule: ScheduleId, metric_name: str = METRIC_NAME
) -> dict[str, str]:
    """Delete a recurring computation of MMD metric."""
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    try:
        request_uuid = uuid.UUID(schedule.requestId)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Invalid request ID format"
        ) from e

    try:
        logger.info("Deleting %s schedule: %s", METRIC_NAME, schedule.requestId)
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


@router.get("/metrics/drift/mmd/requests")
async def list_mmd_requests(
    metric_name: str = METRIC_NAME,
) -> dict[str, list[dict[str, Any]]]:
    """List the currently scheduled computations of MMD metric."""
    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    try:
        requests = scheduler.get_requests(metric_name)

        requests_list = []
        for request_id, request in requests.items():
            if (
                hasattr(request, "model_id")
                and hasattr(request, "batch_size")
                and hasattr(request, "reference_tag")
                and hasattr(request, "fit_columns")
            ):
                requests_list.append(
                    {
                        "id": str(request_id),  # deprecated: use requestId
                        "requestId": str(request_id),
                        "modelId": request.model_id,
                        "metricName": METRIC_NAME,
                        "batchSize": request.batch_size,
                        "referenceTag": request.reference_tag,
                        "fitColumns": request.fit_columns,
                        "numPermutations": getattr(
                            request,
                            "num_permutations",
                            DEFAULT_NUM_PERMUTATIONS,
                        ),
                        "bandwidth": getattr(request, "bandwidth", DEFAULT_BANDWIDTH),
                        "kernel": getattr(request, "kernel", DEFAULT_KERNEL),
                        "alpha": getattr(request, "alpha", DEFAULT_ALPHA),
                        "seed": getattr(request, "seed", None),
                    }
                )
            else:
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
# DEPRECATED: FourierMMD Endpoints (Backward Compatibility)
# ============================================================================
# These endpoints have been renamed to MMD (METRIC_NAME).
# The "FourierMMD" (DEPRECATED_METRIC_NAME) naming was deprecated as the
# metric is now simply called MMD. These endpoints proxy to MMD for backward
# compatibility and may be removed in a future version.
# ============================================================================


class FourierMMDMetricRequest(MMDMetricRequest):
    """DEPRECATED: Use MMDMetricRequest instead.

    Maintained for backward compatibility with old API clients.
    This class inherits from MMDMetricRequest to ensure consistency
    and reduce duplication. All fields and validation behavior are inherited.
    """


@router.post("/metrics/drift/fouriermmd", deprecated=True)
async def compute_fouriermmd(
    request: FourierMMDMetricRequest,
) -> dict[str, float | bool | str | list[str]]:
    """Compute the current value of FourierMMD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/mmd instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await compute_mmd(request)


@router.get("/metrics/drift/fouriermmd/definition", deprecated=True)
async def get_fouriermmd_definition() -> dict[str, str]:
    """Provide a general definition of FourierMMD metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/drift/mmd/definition instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await get_mmd_definition()


@router.post("/metrics/drift/fouriermmd/request", deprecated=True)
async def schedule_fouriermmd(request: FourierMMDMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of FourierMMD metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/drift/mmd/request instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    request.metric_name = DEPRECATED_METRIC_NAME
    return await schedule_mmd(request)


@router.delete("/metrics/drift/fouriermmd/request", deprecated=True)
async def delete_fouriermmd_schedule(schedule: ScheduleId) -> dict[str, str]:
    """Delete a recurring computation of FourierMMD metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/drift/mmd/request instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await delete_mmd_schedule(schedule, metric_name=DEPRECATED_METRIC_NAME)


@router.get("/metrics/drift/fouriermmd/requests", deprecated=True)
async def list_fouriermmd_requests() -> dict[str, list[dict[str, Any]]]:
    """List the currently scheduled computations of FourierMMD metric (deprecated).

    This endpoint is deprecated. Please use
    /metrics/drift/mmd/requests instead.
    """
    log_deprecated_endpoint(logger, DEPRECATED_METRIC_NAME, METRIC_NAME)
    return await list_mmd_requests(metric_name=DEPRECATED_METRIC_NAME)


async def calculate_mmd_metric(
    batch: pd.DataFrame,
    request: BaseMetricRequest,
) -> MetricValueCarrier:
    """Calculate MMD metric for the Prometheus scheduler."""
    data_source = get_data_source()
    reference_df = await data_source.get_dataframe_by_tag(
        request.model_id, request.reference_tag
    )
    fit_columns = request.fit_columns or list(batch.columns)
    alpha = getattr(request, "alpha", DEFAULT_ALPHA)
    n_permutations = getattr(request, "n_permutations", DEFAULT_NUM_PERMUTATIONS)
    kernel = getattr(request, "kernel", DEFAULT_KERNEL)
    bandwidth = getattr(request, "bandwidth", DEFAULT_BANDWIDTH)

    reference_data = reference_df[fit_columns].to_numpy()
    current_data = batch[fit_columns].to_numpy()

    try:
        result = MMD.compute(
            reference_data=reference_data,
            current_data=current_data,
            alpha=alpha,
            n_permutations=n_permutations,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        return MetricValueCarrier(result.get("mmd_value", 0.0))
    except ImportError:
        logger.warning("MMD computation requires 'goodpoints' package")
        return MetricValueCarrier(0.0)


def _register_mmd_calculator() -> None:
    """Register the MMD calculator with the metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register(METRIC_NAME, calculate_mmd_metric)
        scheduler.metrics_directory.register(
            DEPRECATED_METRIC_NAME, calculate_mmd_metric
        )
        logger.info("%s calculator registered with metrics directory", METRIC_NAME)


try:
    _register_mmd_calculator()
except (AttributeError, TypeError) as e:
    logger.warning("Could not register %s calculator on import: %s", METRIC_NAME, e)
