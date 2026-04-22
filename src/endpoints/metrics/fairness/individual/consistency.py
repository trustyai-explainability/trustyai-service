import logging
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.neighbors import NearestNeighbors

from src.service.data.shared_data_source import get_shared_data_source
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.metric_value_carrier import MetricValueCarrier
from src.service.prometheus.shared_prometheus_scheduler import get_shared_prometheus_scheduler

router = APIRouter()
logger = logging.getLogger(__name__)

METRIC_NAME = "IndividualConsistency"
DEFAULT_N_NEIGHBORS = 5
DEFAULT_BATCH_SIZE = 100
CONSISTENCY_FAIRNESS_TARGET = 1.0
DEFAULT_CONSISTENCY_THRESHOLD_DELTA = 0.1


def get_prometheus_scheduler():
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source():
    """Get the shared data source instance."""
    return get_shared_data_source()


class ScheduleId(BaseModel):
    requestId: str


class IndividualConsistencyRequest(BaseMetricRequest):
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    metric_name: str | None = Field(default=None, alias="metricName")
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, alias="batchSize")

    outcome_name: str = Field(alias="outcomeName")
    n_neighbors: int = Field(default=DEFAULT_N_NEIGHBORS, alias="nNeighbors")
    fit_columns: list[str] = Field(default_factory=list, alias="fitColumns")
    threshold_delta: float | None = Field(default=None, alias="thresholdDelta")

    @model_validator(mode="after")
    def _set_default_metric_name(self) -> "IndividualConsistencyRequest":
        if self.metric_name is None:
            self.metric_name = METRIC_NAME
        return self

    def retrieve_tags(self) -> dict[str, str]:
        tags = self.retrieve_default_tags()
        tags["outcomeName"] = self.outcome_name
        tags["nNeighbors"] = str(self.n_neighbors)
        if self.fit_columns:
            tags["fitColumns"] = ",".join(self.fit_columns)
        return tags


def compute_consistency_score(
    features: np.ndarray,
    predictions: np.ndarray,
    n_neighbors: int,
) -> float:
    """Compute individual consistency score (AIF360-compatible).

    For each sample, finds k nearest neighbors and checks prediction consistency.
    Formula: 1 - (1/n) * sum(|y_i - mean(y_neighbors)|)

    Returns a value in [0, 1] where 1.0 = perfect consistency.
    """
    n_samples = len(features)
    if n_samples == 0:
        raise ValueError("Cannot compute consistency on empty data")

    k = min(n_neighbors, n_samples - 1)
    if k < 1:
        raise ValueError(
            f"Need at least 2 samples for consistency computation, got {n_samples}"
        )

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(features)
    _, indices = nbrs.kneighbors(features)

    neighbor_indices = indices[:, 1:]

    consistency = 0.0
    for i in range(n_samples):
        neighbor_preds = predictions[neighbor_indices[i]]
        consistency += abs(predictions[i] - np.mean(neighbor_preds))

    return 1.0 - consistency / n_samples


def calculate_consistency_metric(
    dataframe: pd.DataFrame,
    request: IndividualConsistencyRequest,
) -> MetricValueCarrier:
    """Calculate Individual Consistency for the given dataframe and request.

    Registered with the metrics directory and called by the scheduler.
    """
    outcome_name = request.outcome_name
    fit_columns = request.fit_columns

    if outcome_name not in dataframe.columns:
        raise ValueError(f"Outcome column '{outcome_name}' not found in data")

    if not fit_columns:
        raise ValueError("fitColumns is required — specify feature columns for k-NN distance")

    missing_cols = [c for c in fit_columns if c not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Feature columns not found in data: {missing_cols}")

    features = dataframe[fit_columns].to_numpy().astype(float)
    predictions = dataframe[outcome_name].to_numpy().astype(float)

    if len(features) < 2:
        logger.warning(
            "Insufficient data for consistency calculation: %d samples. Returning NaN.",
            len(features),
        )
        return MetricValueCarrier(float("nan"))

    score = compute_consistency_score(features, predictions, request.n_neighbors)

    logger.debug("Individual Consistency calculation result: %.4f", score)
    return MetricValueCarrier(score)


def register_consistency_calculator() -> None:
    """Register the Individual Consistency calculator with the global metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register(METRIC_NAME, calculate_consistency_metric)
        logger.info("IndividualConsistency calculator registered with metrics directory")


try:
    register_consistency_calculator()
except Exception as e:
    logger.warning("Could not register IndividualConsistency calculator on import: %s", e)


@router.post("/metrics/individual/fairness/consistency")
async def compute_consistency(
    request: IndividualConsistencyRequest,
    delta: float | None = Query(None),
) -> dict:
    """Compute the current value of Individual Consistency metric."""
    try:
        logger.info("Computing Individual Consistency for model: %s", request.model_id)

        data_source = get_data_source()
        batch_size = request.batch_size if request.batch_size else DEFAULT_BATCH_SIZE

        dataframe = await data_source.get_organic_dataframe(request.model_id, batch_size)

        if dataframe.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for model: {request.model_id}",
            )

        result = calculate_consistency_metric(dataframe, request)

        if delta is None:
            delta = (
                request.threshold_delta
                if request.threshold_delta is not None
                else DEFAULT_CONSISTENCY_THRESHOLD_DELTA
            )

        return {
            "name": METRIC_NAME,
            "value": result.get_value(),
            "type": "FAIRNESS",
            "specificDefinition": f"Individual Consistency value of {result.get_value():.4f}",
            "thresholds": {
                "lowerBound": CONSISTENCY_FAIRNESS_TARGET - delta,
                "upperBound": CONSISTENCY_FAIRNESS_TARGET + delta,
                "outsideBounds": result.get_value() < CONSISTENCY_FAIRNESS_TARGET - delta,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error computing Individual Consistency: %s", e)
        raise HTTPException(status_code=500, detail=f"Error computing metric: {e}") from e


@router.get("/metrics/individual/fairness/consistency/definition")
async def get_consistency_definition() -> dict:
    """Provide a general definition of Individual Consistency metric."""
    return {
        "name": "Individual Consistency",
        "description": (
            "Measures consistency of predictions across similar inputs using k-nearest neighbors. "
            "A value of 1.0 indicates perfect consistency (similar inputs always receive the same prediction). "
            "Lower values indicate that the model treats similar individuals differently."
        ),
    }


@router.post("/metrics/individual/fairness/consistency/definition")
async def interpret_consistency_value(request: dict) -> dict:
    """Provide a specific, plain-english interpretation of a consistency value."""
    return {
        "interpretation": "The consistency score indicates how uniformly "
        "the model treats similar individuals."
    }


@router.post("/metrics/individual/fairness/consistency/request")
async def schedule_consistency(request: IndividualConsistencyRequest) -> dict:
    """Schedule a recurring computation of Individual Consistency metric."""
    try:
        request_id = uuid.uuid4()
        logger.info("Scheduling Individual Consistency computation with ID: %s", request_id)

        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        await scheduler.register(METRIC_NAME, request_id, request)

        logger.info(
            "Successfully scheduled Individual Consistency computation with ID: %s",
            request_id,
        )
        return {"requestId": str(request_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error scheduling Individual Consistency computation: %s", e)
        raise HTTPException(status_code=500, detail=f"Error scheduling metric: {e}") from e


@router.delete("/metrics/individual/fairness/consistency/request")
async def delete_consistency_schedule(schedule: ScheduleId) -> dict:
    """Delete a recurring computation of Individual Consistency metric."""
    try:
        logger.info("Deleting Individual Consistency schedule: %s", schedule.requestId)

        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        try:
            request_uuid = uuid.UUID(schedule.requestId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request ID format")

        await scheduler.delete(METRIC_NAME, request_uuid)

        logger.info("Successfully deleted Individual Consistency schedule: %s", schedule.requestId)
        return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting Individual Consistency schedule: %s", e)
        raise HTTPException(status_code=500, detail=f"Error deleting schedule: {e}")


@router.get("/metrics/individual/fairness/consistency/requests")
async def list_consistency_requests() -> dict:
    """List the currently scheduled computations of Individual Consistency metric."""
    try:
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        ic_requests = scheduler.get_requests(METRIC_NAME)

        requests_list = []
        for request_id, request in ic_requests.items():
            if (
                hasattr(request, "model_id")
                and hasattr(request, "batch_size")
                and hasattr(request, "outcome_name")
                and hasattr(request, "n_neighbors")
            ):
                requests_list.append({
                    "requestId": str(request_id),
                    "modelId": request.model_id,
                    "metricName": METRIC_NAME,
                    "batchSize": request.batch_size,
                    "outcomeName": request.outcome_name,
                    "nNeighbors": request.n_neighbors,
                    "fitColumns": getattr(request, "fit_columns", []),
                })
            else:
                logger.warning(
                    "Skipping malformed Individual Consistency request %s: missing required attributes",
                    request_id,
                )
                continue

        return {"requests": requests_list}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing Individual Consistency requests: %s", e)
        raise HTTPException(status_code=500, detail=f"Error listing requests: {e}")
