"""Stable endpoint-neutral error mapping tests."""

import pytest

from trustyai_service.service.explainers.local.error_mapping import map_error
from trustyai_service.service.explainers.local.model_provider import (
    DependencyUnavailableError,
    LocalDataError,
    LocalExecutionError,
    ProviderConfigurationError,
    ProviderDeadlineError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (DependencyUnavailableError(), 503),
        (ProviderUnavailableError(), 503),
        (ProviderDeadlineError(), 504),
        (ProviderInvalidRequestError(), 400),
        (ProviderInvalidResponseError(), 502),
        (ProviderConfigurationError(), 500),
        (LocalDataError("invalid stored data"), 400),
        (LocalExecutionError("training failed"), 500),
    ],
)
def test_provider_error_codes_map_to_stable_statuses(
    error: Exception, status: int
) -> None:
    assert map_error(error).status_code == status
