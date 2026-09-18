"""Request-contract regression tests for local explainers."""

import pytest
from pydantic import ValidationError

from trustyai_service.endpoints.explainers.local_explainer import (
    LocalExplanationModelConfig,
    ModelConfig,
)
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType


def test_model_defaults_to_real_model() -> None:
    config = LocalExplanationModelConfig(
        base_url="https://inference.example",
        model_name="credit-model",
        task=TaskType.REGRESSION,
    )
    assert config.prediction_source is PredictionSource.MODEL


def test_surrogate_may_omit_base_url() -> None:
    config = LocalExplanationModelConfig(
        model_name="credit-model",
        prediction_source=PredictionSource.SURROGATE,
        task=TaskType.CLASSIFICATION,
    )
    assert config.base_url is None


@pytest.mark.parametrize(
    "value",
    [
        "ftp://model.example",
        "https://user:" + "pass@model.example",
    ],  # pragma: allowlist secret
)
def test_model_url_must_be_http_without_credentials(value: str) -> None:
    with pytest.raises(ValidationError):
        LocalExplanationModelConfig(
            base_url=value, model_name="credit-model", task=TaskType.REGRESSION
        )


@pytest.mark.parametrize(
    "value", ["", ".", "..", "model/name", "model%name", "model\nname"]
)
def test_model_segments_are_path_safe(value: str) -> None:
    with pytest.raises(ValidationError):
        LocalExplanationModelConfig(
            base_url="http://model.example",
            model_name=value,
            task=TaskType.REGRESSION,
        )


def test_placeholder_model_contract_remains_unchanged() -> None:
    assert ModelConfig(target="regressor", name="legacy").version is None
