import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.core.metrics.drift.kolmogorov_smirnov_streaming import KolmogorovSmirnovStreaming
from src.service.data.shared_data_source import get_shared_data_source
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.shared_prometheus_scheduler import get_shared_prometheus_scheduler

router = APIRouter()
logger = logging.getLogger(__name__)

# Metric name constant
METRIC_NAME = "KSTestStreaming"
DEPRECATED_METRIC_NAME = "ApproxKSTest"  # Legacy name for backwards compatibility


def get_prometheus_scheduler():
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source():
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    requestId: str


class ApproxKSTestMetricRequest(BaseMetricRequest):
    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: Optional[str] = Field(default=None, alias="metricName")  # Will be set by endpoint
    request_name: Optional[str] = Field(default=None, alias="requestName")
    batch_size: int = Field(default=100, alias="batchSize")

    # ApproxKSTest-specific fields
    threshold_delta: float = Field(default=0.05, alias="thresholdDelta")
    reference_tag: Optional[str] = Field(default=None, alias="referenceTag")
    fit_columns: List[str] = Field(default_factory=list, alias="fitColumns")

    # Streaming-specific field: epsilon for GK sketch accuracy
    epsilon: float = Field(default=0.01, description="Error parameter for GK sketch (default: 0.01)")

    def retrieve_tags(self) -> Dict[str, str]:
        """Retrieve tags for this ApproxKSTest metric request."""
        tags = self.retrieve_default_tags()
        if self.reference_tag:
            tags["referenceTag"] = self.reference_tag
        if self.fit_columns:
            tags["fitColumns"] = ",".join(self.fit_columns)
        if self.epsilon:
            tags["epsilon"] = str(self.epsilon)
        return tags


@router.post("/metrics/drift/ksteststreaming")
async def compute_ksteststreaming(
    request: ApproxKSTestMetricRequest,
) -> Dict[str, float | bool | str | Dict[str, Dict[str, float]]]:
    """Compute the current value of KS Test Streaming metric."""
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

        # Get epsilon parameter
        epsilon = request.epsilon

        # Calculate approximate KS test for each feature
        alpha = request.threshold_delta

        if request.fit_columns:
            # Multi-feature case: iterate over features
            results = {}
            for feature_name in request.fit_columns:
                if feature_name not in reference_df.columns or feature_name not in current_df.columns:
                    raise HTTPException(status_code=400, detail=f"Feature {feature_name} not found in data")

                reference_data = reference_df[feature_name].to_numpy()
                current_data = current_df[feature_name].to_numpy()

                # Use streaming KS test
                ks = KolmogorovSmirnovStreaming(epsilon=epsilon)
                ks.insert_reference_batch(reference_data)
                ks.insert_current_batch(current_data)

                results[feature_name] = ks.kstest(alpha=alpha)

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
        else:
            raise HTTPException(
                status_code=400, detail="fitColumns is required - specify which features to test for drift"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing {METRIC_NAME}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/ksteststreaming/definition")
async def get_ksteststreaming_definition() -> Dict[str, str]:
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
async def schedule_ksteststreaming(request: ApproxKSTestMetricRequest) -> Dict[str, str]:
    """Schedule a recurring computation of KS Test Streaming metric."""
    try:
        # Generate UUID for this request
        request_id = uuid.uuid4()
        logger.info(f"Scheduling {METRIC_NAME} computation with ID: {request_id}.")

        # Set metric name automatically
        request.metric_name = METRIC_NAME

        # Get the scheduler and register the request
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Register with the scheduler (this will reconcile the request and store it)
        await scheduler.register(request.metric_name, request_id, request)

        logger.info(f"Successfully scheduled {METRIC_NAME} computation with ID: {request_id}")
        return {"requestId": str(request_id)}

    except Exception as e:
        logger.error(f"Error scheduling {METRIC_NAME} computation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scheduling metric: {str(e)}") from e


@router.delete("/metrics/drift/ksteststreaming/request")
async def delete_ksteststreaming_schedule(schedule: ScheduleId) -> Dict[str, str]:
    """Delete a recurring computation of KS Test Streaming metric."""
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

    except Exception as e:
        logger.error(f"Error deleting {METRIC_NAME} schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting schedule: {str(e)}")


@router.get("/metrics/drift/ksteststreaming/requests")
async def list_ksteststreaming_requests() -> Dict[str, List[Dict[str, Any]]]:
    """List the currently scheduled computations of KS Test Streaming metric."""
    try:
        # Get the scheduler and list ApproxKSTest requests
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Get all requests for ApproxKSTest
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
                logger.warning(f"Skipping malformed {METRIC_NAME} request {request_id}: missing required attributes")
                continue

        return {"requests": requests_list}

    except Exception as e:
        logger.error(f"Error listing {METRIC_NAME} requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing requests: {str(e)}")


# ============================================================================
# DEPRECATED ENDPOINTS (ApproxKSTest -> KSTestStreaming)
# ============================================================================


@router.post("/metrics/drift/approxkstest", deprecated=True)
async def compute_approxkstest_deprecated(
    request: ApproxKSTestMetricRequest,
) -> Dict[str, float | bool | str | Dict[str, Dict[str, float]]]:
    """Compute the current value of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming instead.
    """
    logger.warning("Deprecated endpoint /metrics/drift/approxkstest called. Use /metrics/drift/ksteststreaming instead.")
    return await compute_ksteststreaming(request)


@router.get("/metrics/drift/approxkstest/definition", deprecated=True)
async def get_approxkstest_definition_deprecated() -> Dict[str, str]:
    """Provide a general definition of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming/definition instead.
    """
    return await get_ksteststreaming_definition()


@router.post("/metrics/drift/approxkstest/request", deprecated=True)
async def schedule_approxkstest_deprecated(request: ApproxKSTestMetricRequest) -> Dict[str, str]:
    """Schedule a recurring computation of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming/request instead.
    """
    logger.warning("Deprecated endpoint /metrics/drift/approxkstest/request called. Use /metrics/drift/ksteststreaming/request instead.")
    return await schedule_ksteststreaming(request)


@router.delete("/metrics/drift/approxkstest/request", deprecated=True)
async def delete_approxkstest_schedule_deprecated(schedule: ScheduleId) -> Dict[str, str]:
    """Delete a recurring computation of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming/request instead.
    """
    return await delete_ksteststreaming_schedule(schedule)


@router.get("/metrics/drift/approxkstest/requests", deprecated=True)
async def list_approxkstest_requests_deprecated() -> Dict[str, List[Dict[str, Any]]]:
    """List the currently scheduled computations of ApproxKSTest metric (deprecated).

    This endpoint is deprecated. Please use /metrics/drift/ksteststreaming/requests instead.
    """
    return await list_ksteststreaming_requests()
