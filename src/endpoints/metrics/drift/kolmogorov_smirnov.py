import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.core.metrics.drift.kolmogorov_smirnov import KolmogorovSmirnov
from src.service.data.shared_data_source import get_shared_data_source
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.shared_prometheus_scheduler import get_shared_prometheus_scheduler

router = APIRouter()
logger = logging.getLogger(__name__)

# Metric name constant
METRIC_NAME = "KSTest"


def get_prometheus_scheduler():
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source():
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    requestId: str


class KSTestMetricRequest(BaseMetricRequest):
    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: Optional[str] = Field(default=None, alias="metricName")  # Will be set by endpoint
    request_name: Optional[str] = Field(default=None, alias="requestName")
    batch_size: int = Field(default=100, alias="batchSize")

    # KSTest-specific fields
    threshold_delta: float = Field(default=0.05, alias="thresholdDelta")  # Default alpha value
    reference_tag: Optional[str] = Field(default=None, alias="referenceTag")
    fit_columns: List[str] = Field(default_factory=list, alias="fitColumns")

    def retrieve_tags(self) -> Dict[str, str]:
        """Retrieve tags for this KSTest metric request."""
        tags = self.retrieve_default_tags()
        if self.reference_tag:
            tags["referenceTag"] = self.reference_tag
        if self.fit_columns:
            tags["fitColumns"] = ",".join(self.fit_columns)
        return tags


@router.post("/metrics/drift/kstest")
async def compute_kstest(request: KSTestMetricRequest) -> Dict[str, float | bool | str | Dict[str, Dict[str, float]]]:
    """Compute the current value of KSTest metric."""
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

        # Calculate KS test for each feature
        alpha = request.threshold_delta

        if request.fit_columns:
            # Multi-feature case: iterate over features
            results = {}
            for feature_name in request.fit_columns:
                if feature_name not in reference_df.columns or feature_name not in current_df.columns:
                    raise HTTPException(status_code=400, detail=f"Feature {feature_name} not found in data")

                reference_data = reference_df[feature_name].to_numpy()
                current_data = current_df[feature_name].to_numpy()

                results[feature_name] = KolmogorovSmirnov.kstest(
                    reference_data=reference_data, current_data=current_data, alpha=alpha
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
        else:
            raise HTTPException(
                status_code=400, detail="fitColumns is required - specify which features to test for drift"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing {METRIC_NAME}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing metric: {str(e)}")


@router.get("/metrics/drift/kstest/definition")
async def get_kstest_definition() -> Dict[str, str]:
    """Provide a general definition of KSTest metric."""
    description = """The two-sampled Kolmogorov–Smirnov test is a nonparametric statistical test.
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
async def schedule_kstest(request: KSTestMetricRequest) -> Dict[str, str]:
    """Schedule a recurring computation of KSTest metric."""
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


@router.delete("/metrics/drift/kstest/request")
async def delete_kstest_schedule(schedule: ScheduleId) -> Dict[str, str]:
    """Delete a recurring computation of KSTest metric."""
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


@router.get("/metrics/drift/kstest/requests")
async def list_kstest_requests() -> Dict[str, List[Dict[str, Any]]]:
    """List the currently scheduled computations of KSTest metric."""
    try:
        # Get the scheduler and list KSTest requests
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

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
                requests_list.append({
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": METRIC_NAME,
                    "batchSize": request.batch_size,
                    "referenceTag": request.reference_tag,
                    "fitColumns": request.fit_columns,
                })
            else:
                # Log warning for malformed request objects and skip them
                logger.warning(f"Skipping malformed {METRIC_NAME} request {request_id}: missing required attributes")
                continue

        return {"requests": requests_list}

    except Exception as e:
        logger.error(f"Error listing {METRIC_NAME} requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing requests: {str(e)}")
