"""Router registration helpers with feature flag support."""

import logging

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

    :param app: FastAPI application instance.
    :param router: Router to register.
    :param flag: Feature flag name (must match keys in ENDPOINTS dict).
    :param tag: Optional tag for OpenAPI grouping.
    :param prefix: Optional URL prefix (e.g. "/metrics" for legacy endpoints).
    """
    if not ENDPOINTS.get(flag, False):
        logger.debug("Skipping router: flag '%s' is disabled", flag)
        return
    kwargs: dict[str, str | list[str]] = {}
    if tag:
        kwargs["tags"] = [tag]
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

    :param app: FastAPI application instance.
    :param router: Router to register.
    :param group_flag: Group-level feature flag name.
    :param metric_flag: Individual metric feature flag name.
    :param tag: Optional tag for OpenAPI grouping.
    :param prefix: Optional URL prefix (e.g. "/metrics" for legacy endpoints).
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
    kwargs: dict[str, str | list[str]] = {}
    if tag:
        kwargs["tags"] = [tag]
    if prefix:
        kwargs["prefix"] = prefix
    app.include_router(router, **kwargs)  # type: ignore[arg-type]
