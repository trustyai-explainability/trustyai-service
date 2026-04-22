"""Router registration helpers with feature flag support.

Disabled endpoints are excluded from the FastAPI router registry at
startup — they do not appear in /docs and have no runtime cost.
"""

import logging
from typing import Any

from fastapi import APIRouter, FastAPI

from src.service.config.feature_flags import ENDPOINTS

logger = logging.getLogger(__name__)


def register_if_enabled(
    app: FastAPI,
    router: APIRouter,
    flag: str,
    tag: str | None = None,
    prefix: str | None = None,
) -> None:
    """Conditionally register a router if the named feature flag is enabled.

    Args:
        app: FastAPI application instance.
        router: Router to register.
        flag: Feature flag name (must match keys in ENDPOINTS dict).
        tag: Optional tag for OpenAPI grouping.
        prefix: Optional URL prefix (e.g. "/metrics" for legacy endpoints).
    """
    if not ENDPOINTS.get(flag, False):
        logger.debug("Skipping router: flag '%s' is disabled", flag)
        return
    kwargs: dict[str, Any] = {"tags": [tag]} if tag else {}
    if prefix:
        kwargs["prefix"] = prefix
    app.include_router(router, **kwargs)  # type: ignore[arg-type]


def register_if_enabled_with_group(
    app: FastAPI,
    router: APIRouter,
    group_flag: str,
    metric_flag: str,
    tag: str | None = None,
    prefix: str | None = None,
) -> None:
    """Register a router gated by both a group flag and an individual metric flag.

    If the group flag is disabled, the metric is never registered.
    If the group is enabled but the individual metric flag is disabled,
    that metric is skipped while others in the same group still register.

    Example: ENDPOINTS["drift"] controls all drift metrics, while
    ENDPOINTS["drift_jensen_shannon"] controls Jensen-Shannon specifically.
    """
    if not ENDPOINTS.get(group_flag, False):
        logger.debug(
            "Skipping %s: group flag '%s' is disabled",
            metric_flag,
            group_flag,
        )
        return
    if not ENDPOINTS.get(metric_flag, False):
        logger.debug(
            "Skipping %s: metric flag '%s' is disabled",
            metric_flag,
            metric_flag,
        )
        return
    kwargs: dict[str, Any] = {"tags": [tag]} if tag else {}
    if prefix:
        kwargs["prefix"] = prefix
    app.include_router(router, **kwargs)  # type: ignore[arg-type]
