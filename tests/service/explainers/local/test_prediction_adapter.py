"""Tests for the shared provider and surrogate prediction adapter."""

import numpy as np
import pytest

from trustyai_service.core.explainers.local.surrogate import build_surrogate
from trustyai_service.service.explainers.local.model_provider import (
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
)
from trustyai_service.service.explainers.local.prediction_adapter import (
    normalize_predictions,
    prediction_callable,
    selected_class_callable,
    selected_scalar_callable,
)
from trustyai_service.service.explainers.local.types import TaskType


def test_provider_and_rf_regression_outputs_share_one_adapter_boundary() -> None:
    """Normalize provider columns and RF vectors to one finite value per row."""
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]])
    surrogate = build_surrogate(
        inputs,
        np.array([0.0, 1.0, 2.0, 3.0]),
        TaskType.REGRESSION,
    )
    rows = np.array([[0.5], [2.5]])

    provider_prediction = prediction_callable(
        lambda values: values[:, :1] + 0.25,
        TaskType.REGRESSION,
    )(rows)
    surrogate_prediction = prediction_callable(surrogate.predict, TaskType.REGRESSION)(
        rows
    )

    assert provider_prediction.shape == (2,)
    assert surrogate_prediction.shape == (2,)
    assert np.isfinite(provider_prediction).all()
    assert np.isfinite(surrogate_prediction).all()


def test_provider_and_rf_classification_outputs_share_lime_adapter_boundary() -> None:
    """Normalize both raw sources to finite multi-class probability matrices."""
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]])
    surrogate = build_surrogate(
        inputs,
        np.array([0, 1, 0, 1]),
        TaskType.CLASSIFICATION,
    )
    rows = np.array([[0.5], [2.5]])

    provider_prediction = prediction_callable(
        lambda values: np.column_stack((1.0 - values[:, 0] / 4.0, values[:, 0] / 4.0)),
        TaskType.CLASSIFICATION,
    )(rows)
    surrogate_prediction = prediction_callable(
        surrogate.predict_proba,
        TaskType.CLASSIFICATION,
    )(rows)

    for prediction in (provider_prediction, surrogate_prediction):
        assert prediction.ndim == 2
        assert prediction.shape == (2, 2)
        assert np.isfinite(prediction).all()
        np.testing.assert_allclose(prediction.sum(axis=1), 1.0)


def test_binary_classification_defaults_shap_selection_to_class_one() -> None:
    """Select the positive class for a binary matrix when no index is supplied."""
    predict = prediction_callable(
        lambda values: np.column_stack(
            (0.8 - values[:, 0] * 0.1, 0.2 + values[:, 0] * 0.1)
        ),
        TaskType.CLASSIFICATION,
    )

    selected = selected_class_callable(predict)

    np.testing.assert_allclose(selected(np.array([[0.0], [1.0]])), [0.2, 0.3])


def test_wider_classification_requires_explicit_shap_class_selection() -> None:
    """Do not guess a class for a probability matrix wider than binary."""
    predict = prediction_callable(
        lambda values: np.tile(np.array([[0.2, 0.3, 0.5]]), (len(values), 1)),
        TaskType.CLASSIFICATION,
    )

    with pytest.raises(ProviderInvalidRequestError, match="class_index"):
        selected_class_callable(predict)(np.ones((1, 1)))

    selected = selected_class_callable(predict, class_index=2)
    np.testing.assert_allclose(selected(np.ones((1, 1))), [0.5])


def test_single_probability_requires_shap_opt_in_and_positive_class() -> None:
    """Allow a one-column positive probability only with explicit SHAP opt-in."""
    without_opt_in = prediction_callable(
        lambda values: np.full((len(values), 1), 0.75),
        TaskType.CLASSIFICATION,
    )
    with pytest.raises(ProviderInvalidResponseError):
        without_opt_in(np.ones((1, 1)))

    with_opt_in = prediction_callable(
        lambda values: np.full((len(values), 1), 0.75),
        TaskType.CLASSIFICATION,
        allow_single_probability=True,
    )
    selected = selected_class_callable(
        with_opt_in,
        single_probability=True,
    )
    np.testing.assert_allclose(selected(np.ones((2, 1))), [0.75, 0.75])

    with pytest.raises(ProviderInvalidRequestError, match="class_index"):
        selected_class_callable(
            with_opt_in,
            class_index=0,
            single_probability=True,
        )(np.ones((1, 1)))


@pytest.mark.parametrize(
    ("raw", "rows"),
    [
        (0.75, 1),
        (np.array([0.75]), 1),
        (np.array([0.25, 0.75]), 2),
    ],
)
def test_classification_rejects_scalar_and_vector_outputs_without_shap_opt_in(
    raw: object,
    rows: int,
) -> None:
    """Require LIME classification outputs to be a two-dimensional matrix."""
    with pytest.raises(ProviderInvalidResponseError):
        normalize_predictions(raw, TaskType.CLASSIFICATION, rows)


def test_shap_single_probability_opt_in_accepts_one_dimensional_positive_output() -> (
    None
):
    """Keep the explicit SHAP opt-in for one-dimensional positive probabilities."""
    predict = prediction_callable(
        lambda values: np.full(len(values), 0.75),
        TaskType.CLASSIFICATION,
        allow_single_probability=True,
    )

    normalized = predict(np.ones((2, 1)))
    assert normalized.shape == (2, 1)
    np.testing.assert_allclose(normalized, [[0.75], [0.75]])

    selected = selected_class_callable(predict, single_probability=True)
    np.testing.assert_allclose(selected(np.ones((2, 1))), [0.75, 0.75])


def test_shap_logit_validates_strict_probability_domain_without_transforming() -> None:
    """Keep raw probabilities unchanged and reject both LOGIT endpoints."""
    predict = prediction_callable(
        lambda values: np.column_stack((1.0 - values[:, 0], values[:, 0])),
        TaskType.CLASSIFICATION,
    )
    selected = selected_class_callable(predict, link="LOGIT")

    np.testing.assert_allclose(selected(np.array([[0.2], [0.8]])), [0.2, 0.8])

    with pytest.raises(ProviderInvalidResponseError, match="strictly inside"):
        selected(np.array([[0.0], [1.0]]))


def test_adapter_rejects_invalid_regression_and_probability_outputs() -> None:
    """Reject malformed, non-finite, non-probability, and multi-output values."""
    with pytest.raises(ProviderInvalidResponseError):
        normalize_predictions([[1.0, 2.0]], TaskType.REGRESSION, 1)
    with pytest.raises(ProviderInvalidResponseError):
        normalize_predictions([[np.nan]], TaskType.REGRESSION, 1)
    with pytest.raises(ProviderInvalidResponseError):
        normalize_predictions([[0.2, 0.2]], TaskType.CLASSIFICATION, 1)
    with pytest.raises(ProviderInvalidResponseError):
        normalize_predictions([[1.2, -0.2]], TaskType.CLASSIFICATION, 1)
    with pytest.raises(ProviderInvalidResponseError):
        normalize_predictions(["not numeric"], TaskType.REGRESSION, 1)


def test_selected_scalar_validates_logit_domain_without_transforming() -> None:
    """Apply the same strict LOGIT input check to a scalar SHAP output."""
    selected = selected_scalar_callable(
        lambda values: np.array([[0.25], [0.75]])[: len(values)],
        link="LOGIT",
    )

    np.testing.assert_allclose(selected(np.ones((2, 1))), [0.25, 0.75])

    invalid = selected_scalar_callable(
        lambda values: np.ones((len(values), 1)),
        link="LOGIT",
    )
    with pytest.raises(ProviderInvalidResponseError, match="strictly inside"):
        invalid(np.ones((1, 1)))
