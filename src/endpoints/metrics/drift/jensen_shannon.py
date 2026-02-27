import logging
import uuid
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.metrics.drift.jensen_shannon import (
    DEFAULT_BINS,
    DEFAULT_GRID_POINTS,
    DEFAULT_METHOD,
    DEFAULT_STATISTIC,
    DEFAULT_THRESHOLD,
    JensenShannon,
)
from src.service.data.shared_data_source import DataSource, get_shared_data_source
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.prometheus_scheduler import PrometheusScheduler
from src.service.prometheus.shared_prometheus_scheduler import get_shared_prometheus_scheduler

router = APIRouter()
logger = logging.getLogger(__name__)

# Metric name constant
METRIC_NAME = "JensenShannon"


def get_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source() -> DataSource:
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    requestId: str


class JensenShannonMetricRequest(BaseMetricRequest):
    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: str | None = Field(default=None, alias="metricName")
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int = Field(default=100, alias="batchSize")

    # JensenShannon-specific fields
    statistic: str = Field(default=DEFAULT_STATISTIC, alias="statistic")  # JS distance or divergence
    threshold: float = Field(default=DEFAULT_THRESHOLD, alias="threshold")  # Drift detection threshold
    method: str = Field(default=DEFAULT_METHOD, alias="method")  # Density estimation method
    grid_points: int = Field(default=DEFAULT_GRID_POINTS, alias="gridPoints")  # Grid points for KDE
    bins: int = Field(default=DEFAULT_BINS, alias="bins")  # Number of bins for histogram method
    reference_tag: str | None = Field(default=None, alias="referenceTag")
    fit_columns: list[str] = Field(default_factory=list, alias="fitColumns")

    @model_validator(mode="after")
    def _set_default_metric_name(self) -> "JensenShannonMetricRequest":
        """Automatically set metric_name to default if not provided."""
        if self.metric_name is None:
            self.metric_name = METRIC_NAME
        return self

    def retrieve_tags(self) -> dict[str, str]:
        """Retrieve tags for this JensenShannon metric request."""
        tags = self.retrieve_default_tags()
        if self.reference_tag:
            tags["referenceTag"] = self.reference_tag
        if self.fit_columns:
            tags["fitColumns"] = ",".join(self.fit_columns)
        return tags


@router.post("/metrics/drift/jensenshannon")
async def compute_jensenshannon(
    request: JensenShannonMetricRequest,
) -> dict[str, float | bool | str | dict[str, dict[str, float | bool]]]:
    """Compute the current value of Jensen-Shannon metric."""
    try:
        logger.info(f"Computing {METRIC_NAME} for model: {request.model_id}")

        # Get data source
        data_source = get_data_source()
        batch_size = request.batch_size

        # Get reference dataframe (tagged with referenceTag)
        if request.reference_tag:
            reference_df = await data_source.get_dataframe_by_tag(request.model_id, request.reference_tag)
        else:
            raise HTTPException(status_code=400, detail="referenceTag is required for drift detection")

        # Get current dataframe (most recent organic data)
        current_df = await data_source.get_organic_dataframe(request.model_id, batch_size)

        if len(reference_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No reference data found for model: {request.model_id} with tag: {request.reference_tag}",
            )

        if len(current_df) == 0:
            raise HTTPException(status_code=404, detail=f"No current data found for model: {request.model_id}")

        # Calculate Jensen-Shannon divergence for each feature
        statistic = request.statistic
        threshold = request.threshold
        method = request.method
        grid_points = request.grid_points
        bins = request.bins

        if request.fit_columns:
            # Multi-feature case: iterate over features
            results = {}
            for feature_name in request.fit_columns:
                if feature_name not in reference_df.columns or feature_name not in current_df.columns:
                    raise HTTPException(status_code=400, detail=f"Feature {feature_name} not found in data")

                reference_data = reference_df[feature_name].to_numpy()
                current_data = current_df[feature_name].to_numpy()

                results[feature_name] = JensenShannon.jensenshannon(
                    data_ref=reference_data,
                    data_cur=current_data,
                    statistic=cast(Literal["distance", "divergence"], statistic),
                    threshold=threshold,
                    method=cast(Literal["kde", "hist"], method),
                    grid_points=grid_points,
                    bins=bins,
                )

            # Aggregate: drift detected if any feature shows drift
            drift_detected = any(r["drift_detected"] for r in results.values())
            max_distance = max(r["Jensen-Shannon_distance"] for r in results.values())
            max_divergence = max(r["Jensen-Shannon_divergence"] for r in results.values())

            return {
                "status": "success",
                "value": max_distance,
                "drift_detected": drift_detected,
                "Jensen-Shannon_distance": max_distance,
                "Jensen-Shannon_divergence": max_divergence,
                "threshold": threshold,
                "statistic": statistic,
                "method": method,
                "feature_results": results,
            }
        else:
            raise HTTPException(
                status_code=400, detail="fitColumns is required - specify which features to test for drift"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing {METRIC_NAME}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/jensenshannon/definition")
async def get_jensenshannon_definition() -> dict[str, str]:
    """Provide a general definition of Jensen-Shannon metric."""
    description = """The Jensen-Shannon divergence is a symmetric
    and smoothed version of the Kullback–Leibler divergence.
    It measures the similarity between two probability distributions
    and is bounded between 0 and 1 (or 0 and log(2) in nats).
    The Jensen-Shannon distance is the square root of the divergence.

    For more information, see the following:
    1. https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    """

    return {
        "name": "Jensen-Shannon Divergence",
        "description": description,
    }


@router.post("/metrics/drift/jensenshannon/request")
async def schedule_jensenshannon(request: JensenShannonMetricRequest) -> dict[str, str]:
    """Schedule a recurring computation of Jensen-Shannon metric."""
    try:
        # Generate UUID for this request
        request_id = uuid.uuid4()
        logger.info(f"Scheduling {METRIC_NAME} computation with ID: {request_id}.")

        # Metric name is automatically set by model_validator
        # Get the scheduler and register the request
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Register with the scheduler (this will reconcile the request and store it)
        # metric_name is guaranteed to be set by model_validator
        assert request.metric_name is not None, "metric_name should be set by model_validator"
        await scheduler.register(request.metric_name, request_id, request)

        logger.info(f"Successfully scheduled {METRIC_NAME} computation with ID: {request_id}")
        return {"requestId": str(request_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling {METRIC_NAME} computation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scheduling metric: {str(e)}") from e


@router.delete("/metrics/drift/jensenshannon/request")
async def delete_jensenshannon_schedule(schedule: ScheduleId) -> dict[str, str]:
    """Delete a recurring computation of Jensen-Shannon metric."""
    try:
        logger.info(f"Deleting {METRIC_NAME} schedule: {schedule.requestId}")

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
        await scheduler.delete(METRIC_NAME, request_uuid)

        logger.info(f"Successfully deleted {METRIC_NAME} schedule: {schedule.requestId}")
        return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting {METRIC_NAME} schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting schedule: {str(e)}")


@router.get("/metrics/drift/jensenshannon/requests")
async def list_jensenshannon_requests() -> dict[str, list[dict[str, Any]]]:
    """List the currently scheduled computations of Jensen-Shannon metric."""
    try:
        # Get the scheduler and list Jensen-Shannon requests
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Get all requests for JensenShannon
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
                requests_list.append({
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": METRIC_NAME,
                    "batchSize": request.batch_size,
                    "referenceTag": request.reference_tag,
                    "fitColumns": request.fit_columns,
                    "statistic": getattr(request, "statistic", "distance"),
                    "threshold": getattr(request, "threshold", 0.1),
                    "method": getattr(request, "method", "kde"),
                    "gridPoints": getattr(request, "grid_points", 256),
                    "bins": getattr(request, "bins", 64),
                })
            else:
                # Log warning for malformed request objects and skip them
                logger.warning(f"Skipping malformed {METRIC_NAME} request {request_id}: missing required attributes")
                continue

        return {"requests": requests_list}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing {METRIC_NAME} requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing requests: {str(e)}")
