"""Tests for the shared local model contract and prediction adapter."""

import numpy as np
import pytest

from trustyai_service.endpoints.explainers.local_explainer import (
    LocalExplanationModelConfig,
)
from trustyai_service.service.explainers.local.kserve_v2_http import normalize_base_url
from trustyai_service.service.explainers.local.model_provider import (
    ProviderInvalidRequestError,
)
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType


def test_model_source_requires_explicit_http_base_url() -> None:
    """Require an explicit HTTP model endpoint for real-model execution."""
    config = LocalExplanationModelConfig(
        base_url="https://inference.example",
        model_name="credit-model",
        task=TaskType.REGRESSION,
    )
    assert config.prediction_source is PredictionSource.MODEL


def test_surrogate_source_can_omit_base_url() -> None:
    """Permit provider-free surrogate execution without an endpoint URL."""
    config = LocalExplanationModelConfig(
        model_name="credit-model",
        prediction_source=PredictionSource.SURROGATE,
        task=TaskType.CLASSIFICATION,
    )
    assert config.base_url is None


@pytest.mark.parametrize(
    "value",
    [
        "model:8080",
        "ftp://inference.example",
        "https://user:" + "pass@example",
    ],  # pragma: allowlist secret
)
def test_base_url_rejects_non_http_or_credentials(value: str) -> None:
    """Reject non-HTTP URLs and URLs containing user credentials."""
    with pytest.raises(ProviderInvalidRequestError, match="base_url"):
        normalize_base_url(value)


def test_model_name_rejects_path_injection() -> None:
    """Reject model names that contain path traversal components."""
    with pytest.raises(ValueError, match="model_name"):
        LocalExplanationModelConfig(
            base_url="http://inference.example",
            model_name="../other",
            task=TaskType.REGRESSION,
        )


def test_model_contract_does_not_infer_task() -> None:
    """Require callers to declare the model task explicitly."""
    with pytest.raises(ValueError, match="task"):
        LocalExplanationModelConfig.model_validate(
            {"base_url": "http://inference.example", "model_name": "m"}
        )


def test_numeric_contract_fixture_is_stable() -> None:
    """Keep the test fixture shape representative of a tabular model input."""
    assert np.asarray([[1.0, 2.0]]).shape == (1, 2)
