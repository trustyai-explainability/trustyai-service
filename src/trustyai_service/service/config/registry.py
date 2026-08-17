"""Router registration helpers with feature flag support."""

import logging

from fastapi import APIRouter, FastAPI

from trustyai_service.service.config import feature_flags

logger = logging.getLogger(__name__)


def _is_enabled(flag: str) -> bool:
    if flag not in feature_flags.ENDPOINTS:
        logger.warning("Unknown feature flag '%s' queried; treating as disabled", flag)
        return False
    return feature_flags.ENDPOINTS[flag]


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
        logger.info("Skipping router: flag '%s' is disabled", flag)
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


def register_with_legacy_prefix(
    app: FastAPI,
    router: APIRouter,
    group_flag: str,
    metric_flag: str,
    modern_tag: str,
    legacy_tag: str,
) -> None:
    """Register a router twice: once modern, once with /metrics prefix.

    Used for endpoints that must support both the current API structure
    and deprecated /metrics-prefixed routes for backwards compatibility.

    :param app: FastAPI application instance.
    :param router: Router to register.
    :param group_flag: Group-level feature flag name.
    :param metric_flag: Individual metric feature flag name.
    :param modern_tag: OpenAPI tag for the modern route.
    :param legacy_tag: OpenAPI tag for the legacy /metrics route.
    """
    # Register modern route
    register_if_enabled_with_group(
        app,
        router,
        group_flag,
        metric_flag,
        tag=modern_tag,
    )
    # Register deprecated /metrics route
    register_if_enabled_with_group(
        app,
        router,
        group_flag,
        metric_flag,
        tag=legacy_tag,
        prefix="/metrics",
    )
