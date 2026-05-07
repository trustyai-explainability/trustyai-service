"""Router registration helpers with feature flag support."""

import logging

from fastapi import APIRouter, FastAPI

from src.service.config.feature_flags import ENDPOINTS

logger = logging.getLogger(__name__)


def _is_enabled(flag: str) -> bool:
    if flag not in ENDPOINTS:
        logger.warning("Unknown feature flag '%s' queried; treating as disabled", flag)
        return False
    return ENDPOINTS[flag]


def _include_router(
    app: FastAPI,
    router: APIRouter,
    tag: str | None = None,
    prefix: str | None = None,
) -> None:
    if tag and prefix:
        app.include_router(router, tags=[tag], prefix=prefix)
    elif tag:
        app.include_router(router, tags=[tag])
    elif prefix:
        app.include_router(router, prefix=prefix)
    else:
        app.include_router(router)


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
    if not _is_enabled(flag):
        logger.debug("Skipping router: flag '%s' is disabled", flag)
        return
    _include_router(app, router, tag, prefix)


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
    if not _is_enabled(group_flag):
        logger.debug(
            "Skipping %s: group flag '%s' is disabled",
            metric_flag,
            group_flag,
        )
        return
    if not _is_enabled(metric_flag):
        logger.debug(
            "Skipping %s: metric flag '%s' is disabled",
            metric_flag,
            metric_flag,
        )
        return
    _include_router(app, router, tag, prefix)
