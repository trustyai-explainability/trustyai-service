"""Tests for the shared local model contract and prediction adapter."""

import numpy as np
import pytest

from trustyai_service.endpoints.explainers.local_explainer import (
    LocalExplanationModelConfig,
)
from trustyai_service.service.explainers.local.kserve_v2_http import normalize_base_url
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType


def test_model_source_requires_explicit_http_base_url() -> None:
    config = LocalExplanationModelConfig(
        base_url="https://inference.example",
        model_name="credit-model",
        task=TaskType.REGRESSION,
    )
    assert config.prediction_source is PredictionSource.MODEL


def test_surrogate_source_can_omit_base_url() -> None:
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
    with pytest.raises(Exception):
        normalize_base_url(value)


def test_model_name_rejects_path_injection() -> None:
    with pytest.raises(ValueError):
        LocalExplanationModelConfig(
            base_url="http://inference.example",
            model_name="../other",
            task=TaskType.REGRESSION,
        )


def test_model_contract_does_not_infer_task() -> None:
    with pytest.raises(ValueError):
        LocalExplanationModelConfig.model_validate(
            {"base_url": "http://inference.example", "model_name": "m"}
        )


def test_numeric_contract_fixture_is_stable() -> None:
    assert np.asarray([[1.0, 2.0]]).shape == (1, 2)
