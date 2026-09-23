"""Tests for the protocol-neutral local model provider contract."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from trustyai_service.service.explainers.local.model_provider import (
    DependencyUnavailableError,
    PredictionMetadata,
    PredictionProvider,
    ProviderConfigurationError,
    ProviderDeadlineError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType

if TYPE_CHECKING:
    from collections.abc import Callable


def _metadata() -> PredictionMetadata:
    return PredictionMetadata(
        input_name="features",
        output_name="prediction",
        input_datatype="FP32",
        output_datatype="FP32",
        input_shape=(-1, 3),
        output_shape=(-1, 1),
    )


def test_prediction_source_and_task_type_use_stable_values() -> None:
    """Keep the public enum values stable for request and service boundaries."""
    assert PredictionSource.MODEL.value == "MODEL"
    assert PredictionSource.SURROGATE.value == "SURROGATE"
    assert TaskType.REGRESSION.value == "REGRESSION"
    assert TaskType.CLASSIFICATION.value == "CLASSIFICATION"


def test_prediction_metadata_is_immutable() -> None:
    """Prevent callers from changing negotiated tensor metadata in place."""
    metadata = _metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.input_name = "other"  # type: ignore[assignment]


class FakePredictionProvider:
    """Minimal provider used to exercise the structural protocol contract."""

    def __init__(self) -> None:
        """Initialize the fake provider with stable metadata and close tracking."""
        self._metadata = _metadata()
        self.close_calls = 0

    @property
    def metadata(self) -> PredictionMetadata:
        """Return the fake provider metadata."""
        return self._metadata

    def predict(
        self,
        inputs: np.ndarray,
        *,
        timeout_seconds: float | None = None,
    ) -> np.ndarray:
        """Return the first input column as a deterministic fake prediction."""
        del timeout_seconds
        return np.asarray(inputs[:, :1])

    def close(self) -> None:
        """Record the first close call and ignore repeated cleanup."""
        if self.close_calls == 0:
            self.close_calls += 1


def test_fake_provider_conforms_and_close_is_idempotent() -> None:
    """Accept a structural provider and allow cleanup to be called twice."""
    provider: PredictionProvider = FakePredictionProvider()

    assert isinstance(provider, PredictionProvider)
    np.testing.assert_array_equal(
        provider.predict(np.asarray([[1.0, 2.0, 3.0]]), timeout_seconds=1.0),
        np.asarray([[1.0]]),
    )

    provider.close()
    provider.close()
    assert provider.close_calls == 1


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (DependencyUnavailableError, "dependency_unavailable"),
        (ProviderConfigurationError, "configuration_invalid"),
        (ProviderInvalidRequestError, "invalid_request"),
        (ProviderUnavailableError, "unavailable"),
        (ProviderInvalidResponseError, "invalid_response"),
        (ProviderUnsupportedModelError, "unsupported_model"),
        (ProviderDeadlineError, "deadline_exceeded"),
    ],
)
def test_provider_errors_expose_stable_codes(
    error_type: Callable[[str], ProviderError], code: str
) -> None:
    """Expose the stable code while preserving a caller-provided message."""
    error = error_type("test failure")

    assert isinstance(error, ProviderError)
    assert error.code == code
    assert str(error) == "test failure"


def test_provider_modules_are_safe_to_import_without_optional_integrations() -> None:
    """Import the provider contract without loading optional integrations."""
    repository_root = Path(__file__).parents[4]
    script = """
import builtins
import sys

sys.path.insert(0, "src")
forbidden = (
    "httpx2",
    "lime",
    "shap",
    "fastapi",
    "trustyai_service.service.data.storage",
    "trustyai_service.service.data.shared_data_source",
)
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == module or name.startswith(module + ".") for module in forbidden):
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import trustyai_service.service.explainers.local.model_provider  # noqa: E402
import trustyai_service.service.explainers.local.types  # noqa: E402
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
