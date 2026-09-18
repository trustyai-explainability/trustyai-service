"""Shared prediction-adapter contract tests."""

import numpy as np
import pytest

from trustyai_service.service.explainers.local.model_provider import (
    ProviderInvalidResponseError,
)
from trustyai_service.service.explainers.local.prediction_adapter import (
    normalize_predictions,
    prediction_callable,
    selected_class_callable,
    selected_scalar_callable,
)
from trustyai_service.service.explainers.local.types import TaskType


def test_regression_normalizes_single_output_column() -> None:
    result = normalize_predictions([[1.0], [2.0]], TaskType.REGRESSION, 2)
    np.testing.assert_allclose(result, [1.0, 2.0])


def test_classification_normalizes_binary_probability_vector() -> None:
    result = normalize_predictions([0.2, 0.8], TaskType.CLASSIFICATION, 2)
    np.testing.assert_allclose(result, [[0.8, 0.2], [0.2, 0.8]])


def test_classification_rejects_bad_probability_rows() -> None:
    with pytest.raises(ProviderInvalidResponseError):
        normalize_predictions([[0.2, 0.2]], TaskType.CLASSIFICATION, 1)


def test_single_probability_requires_explicit_opt_in() -> None:
    with pytest.raises(ProviderInvalidResponseError):
        normalize_predictions([[0.8]], TaskType.CLASSIFICATION, 1)
    result = normalize_predictions(
        [[0.8]], TaskType.CLASSIFICATION, 1, allow_single_probability=True
    )
    np.testing.assert_allclose(result, [[0.8]])


def test_selected_class_callable_returns_one_scalar_per_row() -> None:
    predict = prediction_callable(
        lambda values: np.column_stack((values[:, 0], 1 - values[:, 0])),
        TaskType.CLASSIFICATION,
    )
    selected = selected_class_callable(predict, 1)
    np.testing.assert_allclose(selected(np.array([[0.2], [0.7]])), [0.8, 0.3])


def test_adapter_rejects_non_numeric_provider_output() -> None:
    with pytest.raises(ProviderInvalidResponseError):
        normalize_predictions(["bad"], TaskType.REGRESSION, 1)


def test_shap_class_adapter_validates_link_and_single_probability() -> None:
    single = selected_class_callable(
        lambda values: np.full((len(values), 1), 0.5),
        1,
        link="LOGIT",
        single_probability=True,
    )
    np.testing.assert_allclose(single(np.ones((2, 1))), [0.5, 0.5])
    invalid = selected_class_callable(
        lambda values: np.ones((len(values), 2)), 1, link="LOGIT"
    )
    with pytest.raises(ProviderInvalidResponseError):
        invalid(np.ones((1, 1)))


def test_shap_scalar_adapter_validates_link() -> None:
    selected = selected_scalar_callable(
        lambda values: np.ones((len(values), 1)), link="LOGIT"
    )
    with pytest.raises(ProviderInvalidResponseError):
        selected(np.ones((1, 1)))
