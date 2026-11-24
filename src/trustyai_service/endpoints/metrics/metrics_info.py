"""Metrics information endpoint for retrieving scheduled metric computations."""

import logging
import uuid
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from trustyai_service.endpoints import routes
from trustyai_service.service.payloads.metrics.base_metric_request import (
    BaseMetricRequest,
)
from trustyai_service.service.prometheus.prometheus_scheduler import (
    PrometheusScheduler,
)
from trustyai_service.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Metric names as registered by each endpoint module. Kept as explicit sets so
# the type filter does not import endpoint modules (and their registration
# side effects) just to read a constant.
DRIFT_METRICS = frozenset(
    {
        "CompareMeans",
        "Meanshift",
        "KSTest",
        "KSTESTSTREAMING",
        "APPROXKSTEST",
        "JensenShannon",
        "MMD",
        "FOURIERMMD",
    }
)
FAIRNESS_METRICS = frozenset({"SPD", "DIR"})
METRIC_TYPES: dict[str, frozenset[str]] = {
    "drift": DRIFT_METRICS,
    "fairness": FAIRNESS_METRICS,
}


class ScheduledMetricRequest(BaseModel):
    """One scheduled metric computation in the Java-compatible response shape."""

    id: str
    request: dict[str, Any]


class ScheduledMetricRequestList(BaseModel):
    """All scheduled metric computations."""

    requests: list[ScheduledMetricRequest]


def get_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def _select_metric_names(metric_type: str | None) -> frozenset[str] | None:
    """Return the metric names selected by the ``type`` filter, or None for all."""
    if metric_type is None or metric_type.lower() == "all":
        return None
    try:
        return METRIC_TYPES[metric_type.lower()]
    except KeyError:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=(
                f"Invalid metric requests type: {metric_type}. "
                f"Expected one of: all, {', '.join(sorted(METRIC_TYPES))}"
            ),
        ) from None


def _serialize_request(
    request_id: uuid.UUID,
    metric_name: str,
    request: BaseMetricRequest,
) -> ScheduledMetricRequest | None:
    """Return the re-POSTable payload for a scheduled request."""
    if not isinstance(request, BaseMetricRequest):
        logger.warning(
            "Skipping malformed %s request %s: not a BaseMetricRequest",
            metric_name,
            request_id,
        )
        return None

    payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
    # The metric name is what the scheduling endpoint stored, and clients (for
    # example odh-cli --metrics-skip-existing) key on it.
    payload["metricName"] = request.metric_name or metric_name

    return ScheduledMetricRequest(id=str(request_id), request=payload)


@router.get(routes.METRICS_ALL_REQUESTS)
async def get_all_metric_requests(
    metric_type: Annotated[
        str | None,
        Query(
            alias="type",
            description=(
                "Filter by metric type: 'all' (default), 'drift' or 'fairness'."
            ),
        ),
    ] = None,
) -> ScheduledMetricRequestList:
    """Retrieve a list of all currently scheduled metric computations."""
    selected_metrics = _select_metric_names(metric_type)
    logger.info("Retrieving all metric requests, type filter: %s", metric_type)

    scheduler = get_prometheus_scheduler()
    if not scheduler:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Prometheus scheduler not available",
        )

    try:
        requests_by_metric = scheduler.get_all_requests()
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for scheduler errors
        logger.exception("Error listing all metric requests")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Error listing requests. Check server logs for details.",
        ) from e

    requests_list: list[ScheduledMetricRequest] = []
    for metric_name in sorted(requests_by_metric):
        if selected_metrics is not None and metric_name not in selected_metrics:
            continue
        for request_id, request in sorted(
            requests_by_metric[metric_name].items(),
            key=lambda item: str(item[0]),
        ):
            entry = _serialize_request(request_id, metric_name, request)
            if entry is not None:
                requests_list.append(entry)

    return ScheduledMetricRequestList(requests=requests_list)
