import logging
import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import ConfigDict, Field, model_validator

from src.core.metrics.fairness.fairness_metrics_utils import calculate_confusion_matrix
from src.endpoints.metrics.fairness.group.utils import (
    GroupDefinitionRequest,
    GroupMetricRequest,
    ScheduleId,
    get_data_source,
    get_prometheus_scheduler,
    prepare_fairness_data,
)
from src.service.prometheus.metric_value_carrier import MetricValueCarrier

router = APIRouter()
logger = logging.getLogger(__name__)

METRIC_NAME = "AOD"
DEFAULT_AOD_THRESHOLD_DELTA = 0.1
DEFAULT_BATCH_SIZE = 100
AOD_FAIRNESS_TARGET = 0
EPSILON = 1e-10


class AODMetricRequest(GroupMetricRequest):
    model_config = ConfigDict(populate_by_name=True)

    label_name: str = Field(alias="labelName")

    @model_validator(mode="after")
    def _set_default_metric_name(self) -> "AODMetricRequest":
        if self.metric_name is None:
            self.metric_name = METRIC_NAME
        return self

    def retrieve_tags(self) -> dict[str, str]:
        tags = self.retrieve_default_tags()
        tags["protectedAttribute"] = self.protected_attribute
        tags["outcomeName"] = self.outcome_name
        tags["labelName"] = self.label_name
        return tags


def calculate_aod_metric(dataframe: pd.DataFrame, request: AODMetricRequest) -> MetricValueCarrier:
    """Calculate Average Odds Difference for the given dataframe and request.

    Registered with the metrics directory and called by the scheduler.
    """
    privileged_data, unprivileged_data, outcome_name, favorable_values = prepare_fairness_data(dataframe, request)

    if len(privileged_data) == 0 or len(unprivileged_data) == 0:
        logger.warning(
            "Insufficient data for AOD calculation: privileged=%d, unprivileged=%d samples. Returning NaN.",
            len(privileged_data),
            len(unprivileged_data),
        )
        return MetricValueCarrier(float("nan"))

    if len(favorable_values) == 0:
        raise ValueError("No favorable outcomes specified for AOD calculation")

    label_name = request.label_name

    if label_name not in dataframe.columns:
        raise ValueError(f"Ground truth column '{label_name}' not found in data")

    positive_class = int(favorable_values[0])

    pcm = calculate_confusion_matrix(
        privileged_data[outcome_name].to_numpy(),
        privileged_data[label_name].to_numpy(),
        positive_class,
    )
    ucm = calculate_confusion_matrix(
        unprivileged_data[outcome_name].to_numpy(),
        unprivileged_data[label_name].to_numpy(),
        positive_class,
    )

    utp, utn, ufp, ufn = ucm["tp"], ucm["tn"], ucm["fp"], ucm["fn"]
    ptp, ptn, pfp, pfn = pcm["tp"], pcm["tn"], pcm["fp"], pcm["fn"]

    aod_value = (utp / (utp + ufn + EPSILON) - ptp / (ptp + pfn + EPSILON)) / 2 + (
        ufp / (ufp + utn + EPSILON) - pfp / (pfp + ptn + EPSILON)
    ) / 2

    logger.debug("AOD calculation result: %.4f", aod_value)
    return MetricValueCarrier(aod_value)


def register_aod_calculator() -> None:
    """Register the AOD calculator with the global metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register(METRIC_NAME, calculate_aod_metric)
        logger.info("AOD calculator registered with metrics directory")


try:
    register_aod_calculator()
except Exception as e:
    logger.warning("Could not register AOD calculator on import: %s", e)


@router.post("/metrics/group/fairness/aod")
async def compute_aod(
    request: AODMetricRequest,
    delta: float | None = Query(None),
) -> dict:
    """Compute the current value of Average Odds Difference metric."""
    try:
        logger.info("Computing AOD for model: %s", request.model_id)

        data_source = get_data_source()
        batch_size = request.batch_size if request.batch_size else DEFAULT_BATCH_SIZE

        dataframe = await data_source.get_organic_dataframe(request.model_id, batch_size)

        if dataframe.empty:
            raise HTTPException(status_code=404, detail=f"No data found for model: {request.model_id}")

        result = calculate_aod_metric(dataframe, request)

        if delta is None:
            delta = request.threshold_delta if request.threshold_delta is not None else DEFAULT_AOD_THRESHOLD_DELTA

        return {
            "name": METRIC_NAME,
            "value": result.get_value(),
            "type": "FAIRNESS",
            "specificDefinition": f"Average Odds Difference value of {result.get_value():.4f}",
            "thresholds": {
                "lowerBound": AOD_FAIRNESS_TARGET - delta,
                "upperBound": AOD_FAIRNESS_TARGET + delta,
                "outsideBounds": abs(result.get_value() - AOD_FAIRNESS_TARGET) > delta,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error computing AOD: %s", e)
        raise HTTPException(status_code=500, detail=f"Error computing metric: {e}") from e


@router.get("/metrics/group/fairness/aod/definition")
async def get_aod_definition() -> dict:
    """Provide a general definition of Average Odds Difference metric."""
    return {
        "name": "Average Odds Difference",
        "description": "Measures the average of the differences in true positive rates and false positive rates "
        "between unprivileged and privileged groups. "
        "A value of zero indicates equal odds across groups. "
        "Positive values indicate the unprivileged group has higher TPR/FPR, "
        "negative values indicate the privileged group has higher TPR/FPR.",
    }


@router.post("/metrics/group/fairness/aod/definition")
async def interpret_aod_value(request: GroupDefinitionRequest) -> dict:
    """Provide a specific, plain-english interpretation of a specific value of AOD metric."""
    try:
        logger.info("Interpreting AOD value for model: %s", request.modelId)
        return {
            "interpretation": "The AOD value indicates the average difference in "
            "true positive and false positive rates between groups."
        }
    except Exception as e:
        logger.error("Error interpreting AOD value: %s", e)
        raise HTTPException(status_code=500, detail=f"Error interpreting value: {e}")


@router.post("/metrics/group/fairness/aod/request")
async def schedule_aod(request: AODMetricRequest) -> dict:
    """Schedule a recurring computation of AOD metric."""
    try:
        request_id = uuid.uuid4()
        logger.info("Scheduling AOD computation with ID: %s", request_id)

        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        await scheduler.register(METRIC_NAME, request_id, request)

        logger.info("Successfully scheduled AOD computation with ID: %s", request_id)
        return {"requestId": str(request_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling AOD computation: %s", e)
        raise HTTPException(status_code=500, detail=f"Error scheduling metric: {e}") from e


@router.delete("/metrics/group/fairness/aod/request")
async def delete_aod_schedule(schedule: ScheduleId) -> dict:
    """Delete a recurring computation of AOD metric."""
    try:
        logger.info("Deleting AOD schedule: %s", schedule.requestId)

        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        try:
            request_uuid = uuid.UUID(schedule.requestId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request ID format")

        await scheduler.delete(METRIC_NAME, request_uuid)

        logger.info("Successfully deleted AOD schedule: %s", schedule.requestId)
        return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting AOD schedule: %s", e)
        raise HTTPException(status_code=500, detail=f"Error deleting schedule: {e}")


@router.get("/metrics/group/fairness/aod/requests")
async def list_aod_requests() -> dict:
    """List the currently scheduled computations of AOD metric."""
    try:
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        aod_requests = scheduler.get_requests(METRIC_NAME)

        requests_list = []
        for request_id, request in aod_requests.items():
            if (
                hasattr(request, "model_id")
                and hasattr(request, "batch_size")
                and hasattr(request, "protected_attribute")
                and hasattr(request, "outcome_name")
                and hasattr(request, "label_name")
            ):
                requests_list.append({
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": METRIC_NAME,
                    "batchSize": request.batch_size,
                    "protectedAttribute": request.protected_attribute,
                    "outcomeName": request.outcome_name,
                    "labelName": request.label_name,
                })
            else:
                logger.warning("Skipping malformed AOD request %s: missing required attributes", request_id)
                continue

        return {"requests": requests_list}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing AOD requests: %s", e)
        raise HTTPException(status_code=500, detail=f"Error listing requests: {e}")


# Deprecated AOD endpoints
@router.post("/aod", deprecated=True)
async def compute_aod_deprecated(request: AODMetricRequest, delta: float | None = Query(None)) -> dict:
    """Compute the current value of Average Odds Difference metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/aod instead.
    """
    return await compute_aod(request, delta)


@router.get("/aod/definition", deprecated=True)
async def get_aod_definition_deprecated() -> dict:
    """Provide a general definition of Average Odds Difference metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/aod/definition instead.
    """
    return await get_aod_definition()


@router.post("/aod/definition", deprecated=True)
async def interpret_aod_value_deprecated(request: GroupDefinitionRequest) -> dict:
    """Provide a specific interpretation of an AOD metric value (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/aod/definition instead.
    """
    return await interpret_aod_value(request)


@router.post("/aod/request", deprecated=True)
async def schedule_aod_deprecated(request: AODMetricRequest) -> dict:
    """Schedule a recurring computation of AOD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/aod/request instead.
    """
    return await schedule_aod(request)


@router.delete("/aod/request", deprecated=True)
async def delete_aod_schedule_deprecated(schedule: ScheduleId) -> dict:
    """Delete a recurring computation of AOD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/aod/request instead.
    """
    return await delete_aod_schedule(schedule)


@router.get("/aod/requests", deprecated=True)
async def list_aod_requests_deprecated() -> dict:
    """List the currently scheduled computations of AOD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/aod/requests instead.
    """
    return await list_aod_requests()
