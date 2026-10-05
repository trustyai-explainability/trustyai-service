"""Tests for stable, endpoint-neutral local-explainer error mapping."""

import pytest

from trustyai_service.service.explainers.local.error_mapping import (
    LocalDataError,
    LocalDataNotFoundError,
    LocalExecutionError,
    LocalWorkerCapacityError,
    map_error,
)
from trustyai_service.service.explainers.local.model_provider import (
    DependencyUnavailableError,
    ProviderConfigurationError,
    ProviderDeadlineError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)


class UntrustedDataError(ValueError):
    """An untyped failure carrying a recognized mapping code."""

    code = "data_invalid"


@pytest.mark.parametrize(
    ("error", "code", "status"),
    [
        (LocalDataError("invalid stored data"), "data_invalid", 400),
        (LocalDataNotFoundError("stored data is missing"), "data_missing", 404),
        (ProviderInvalidRequestError(), "invalid_request", 400),
        (DependencyUnavailableError(), "dependency_unavailable", 503),
        (ProviderConfigurationError(), "configuration_invalid", 500),
        (ProviderUnavailableError(), "unavailable", 503),
        (ProviderInvalidResponseError(), "invalid_response", 502),
        (ProviderUnsupportedModelError(), "unsupported_model", 502),
        (ProviderDeadlineError(), "deadline_exceeded", 504),
        (LocalExecutionError("training failed"), "execution_failed", 500),
        (LocalWorkerCapacityError(), "resource_exhausted", 503),
        (ImportError("optional dependency is absent"), "dependency_unavailable", 503),
        (TimeoutError("worker deadline elapsed"), "deadline_exceeded", 504),
    ],
)
def test_map_error_returns_stable_code_and_status(
    error: Exception,
    code: str,
    status: int,
) -> None:
    """Map every boundary category without constructing an HTTP response."""
    mapped = map_error(error)

    assert mapped.code == code
    assert mapped.status_code == status


def test_unrecognized_failures_map_to_generic_internal_error() -> None:
    """Do not expose arbitrary exception details through the shared mapper."""
    mapped = map_error(RuntimeError("implementation detail"))

    assert mapped.code == "execution_failed"
    assert mapped.status_code == 500
    assert mapped.detail == "Local explanation failed"


@pytest.mark.parametrize("error", [ValueError("fit failed"), KeyError("internal")])
def test_untyped_data_like_failures_map_to_safe_internal_error(
    error: Exception,
) -> None:
    """Do not infer public data failures from untyped built-in exceptions."""
    mapped = map_error(error)

    assert mapped.code == "execution_failed"
    assert mapped.status_code == 500
    assert mapped.detail == "Local explanation failed"


def test_untyped_subclass_with_recognized_code_maps_to_safe_internal_error() -> None:
    """Do not trust mapping codes from arbitrary exception subclasses."""
    mapped = map_error(UntrustedDataError("private implementation detail"))

    assert mapped.code == "execution_failed"
    assert mapped.status_code == 500
    assert mapped.detail == "Local explanation failed"


@pytest.mark.parametrize(
    ("error", "code", "status", "detail"),
    [
        (
            LocalDataError("dataset password=do-not-return"),
            "data_invalid",
            400,
            "Local explanation data is invalid",
        ),
        (
            LocalDataNotFoundError("storage service=internal-database"),
            "data_missing",
            404,
            "Local explanation data was not found",
        ),
        (
            ProviderInvalidRequestError("provider service=internal-model"),
            "invalid_request",
            400,
            "Invalid local explanation request",
        ),
    ],
)
def test_typed_errors_use_allowlisted_messages(
    error: Exception,
    code: str,
    status: int,
    detail: str,
) -> None:
    """Keep stable mappings while hiding typed exception details."""
    mapped = map_error(error)

    assert mapped.code == code
    assert mapped.status_code == status
    assert mapped.detail == detail
    assert str(error) not in mapped.detail
    assert mapped.as_http_detail() == {"code": code, "message": detail}
