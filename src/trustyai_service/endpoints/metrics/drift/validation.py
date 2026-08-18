"""Shared validation utilities for drift metric endpoints."""

import logging
from http import HTTPStatus
from typing import Protocol

from fastapi import HTTPException

from trustyai_service.service.data.shared_data_source import get_shared_data_source

logger = logging.getLogger(__name__)


class DriftRequestProtocol(Protocol):
    """Protocol for drift metric requests requiring validation.

    All drift endpoints (compare_means, jensen_shannon, kolmogorov_smirnov,
    kolmogorov_smirnov_streaming, mmd) must have these attributes.
    """

    model_id: str
    reference_tag: str | None
    fit_columns: list[str] | None


async def validate_drift_request(request: DriftRequestProtocol) -> list[str]:
    """Validate common drift request fields and return validated fit_columns.

    Validates both referenceTag and fitColumns with consistent logic across
    all drift endpoints.

    Args:
        request: Drift metric request with model_id, reference_tag, and fit_columns

    Returns:
        list[str]: Validated, non-empty list of feature column names.
                   Also mutates request.fit_columns in-place for consistency.

    Raises:
        HTTPException:
            - HTTP 400 if referenceTag is missing or whitespace-only
            - HTTP 400 if fitColumns is explicitly empty or contains only whitespace


    Behavior for fitColumns:
        - If fit_columns is None (field omitted): auto-derive from model input schema
        - If fit_columns is [] (explicit empty list): raise HTTP 400
        - If fit_columns is provided: validate not empty/whitespace-only, strip whitespace

    """
    # Validate reference tag (required for all drift detection)
    if not request.reference_tag or not request.reference_tag.strip():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="referenceTag is required for drift detection",
        )

    # Validate and derive fit_columns
    if request.fit_columns is None:
        # Field was omitted - auto-derive from metadata
        data_source = get_shared_data_source()
        metadata = await data_source.get_metadata(request.model_id)
        request.fit_columns = list(metadata.input_schema.items.keys())
        logger.info(
            "fitColumns not specified, using all input columns for model %s: %s",
            request.model_id,
            request.fit_columns,
        )
        return request.fit_columns

    if not request.fit_columns:
        # Field was explicitly set to empty list
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="fitColumns must contain at least one non-empty feature name",
        )

    # Validate provided columns are not empty/whitespace
    valid_features = [f.strip() for f in request.fit_columns if f.strip()]
    if not valid_features:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="fitColumns must contain at least one non-empty feature name",
        )

    request.fit_columns = valid_features
    return valid_features
