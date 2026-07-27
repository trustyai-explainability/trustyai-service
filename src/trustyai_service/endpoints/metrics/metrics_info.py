"""Metrics information endpoint for retrieving scheduled metric computations."""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics/all/requests")
async def get_all_metric_requests(type_: str | None = None) -> dict[str, list[Any]]:
    """Retrieve a list of all currently scheduled metric computations."""
    logger.info("Retrieving all metric requests, type filter: %s", type_)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Metric request listing is not yet implemented",
    )
