"""Normalize provider and sklearn outputs to algorithm callables."""

from collections.abc import Callable

import numpy as np

from .model_provider import ProviderInvalidResponseError
from .types import TaskType

_MATRIX_RANK = 2
_MIN_CLASS_COUNT = 2


def normalize_predictions(
    raw: object, task: TaskType, rows: int, *, allow_single_probability: bool = False
) -> np.ndarray:
    """Normalize one provider batch to the task-specific algorithm shape."""
    values = np.asarray(raw)
    if values.ndim == 0 or values.dtype.kind not in "biuf":
        msg = "Predictions must be numeric arrays"
        raise ProviderInvalidResponseError(msg)
    if values.shape[0] != rows:
        msg = "Prediction row count does not match input"
        raise ProviderInvalidResponseError(msg)
    if task is TaskType.REGRESSION:
        return _normalize_regression(values)
    return _normalize_classification(
        values, allow_single_probability=allow_single_probability
    )


def _normalize_regression(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return _finite_float(values, "Regression predictions")
    if values.ndim == _MATRIX_RANK and values.shape[1] == 1:
        return _finite_float(values[:, 0], "Regression predictions")
    msg = "Regression model must return one numeric output"
    raise ProviderInvalidResponseError(msg)


def _normalize_classification(
    values: np.ndarray, *, allow_single_probability: bool
) -> np.ndarray:
    if values.ndim == 1:
        probability = _finite_float(values, "Classification probabilities")
        values = np.column_stack((1.0 - probability, probability))
    elif values.ndim == _MATRIX_RANK and (
        values.shape[1] >= _MIN_CLASS_COUNT
        or (values.shape[1] == 1 and allow_single_probability)
    ):
        values = _finite_float(values, "Classification probabilities")
    else:
        msg = "Classification model must return probabilities"
        raise ProviderInvalidResponseError(msg)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        msg = "Classification probabilities are invalid"
        raise ProviderInvalidResponseError(msg)
    if values.shape[1] > 1 and not np.allclose(
        values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-3
    ):
        msg = "Classification probabilities must sum to one"
        raise ProviderInvalidResponseError(msg)
    return values


def _finite_float(values: np.ndarray, label: str) -> np.ndarray:
    try:
        result = values.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = f"{label} must be numeric"
        raise ProviderInvalidResponseError(msg) from exc
    if not np.isfinite(result).all():
        msg = f"{label} must be finite"
        raise ProviderInvalidResponseError(msg)
    return result


def selected_class_callable(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    class_index: int,
    *,
    link: str = "IDENTITY",
    single_probability: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable selecting and validating one class probability."""

    def predict(values: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(predict_fn(values))
        if probabilities.ndim != _MATRIX_RANK:
            msg = "Classification output must be a matrix"
            raise ProviderInvalidResponseError(msg)
        if probabilities.shape[1] == 1:
            if not single_probability or class_index != 1:
                msg = "Selected class index is invalid"
                raise ProviderInvalidResponseError(msg)
            selected = probabilities[:, 0]
        elif not 0 <= class_index < probabilities.shape[1]:
            msg = "Selected class index is invalid"
            raise ProviderInvalidResponseError(msg)
        else:
            selected = probabilities[:, class_index]
        selected = _finite_float(selected, "Selected class probabilities")
        if link.upper() == "LOGIT" and not np.all((selected > 0) & (selected < 1)):
            msg = "LOGIT SHAP requires probabilities strictly inside (0, 1)"
            raise ProviderInvalidResponseError(msg)
        return selected

    return predict


def selected_scalar_callable(
    predict_fn: Callable[[np.ndarray], np.ndarray], *, link: str = "IDENTITY"
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable selecting and validating one scalar output."""

    def predict(values: np.ndarray) -> np.ndarray:
        raw = np.asarray(predict_fn(values))
        if raw.ndim == _MATRIX_RANK and raw.shape[1] == 1:
            raw = raw[:, 0]
        if raw.ndim != 1 or raw.shape[0] != len(values):
            msg = "Scalar model output is malformed"
            raise ProviderInvalidResponseError(msg)
        selected = _finite_float(raw, "Scalar model output")
        if link.upper() == "LOGIT" and not np.all((selected > 0) & (selected < 1)):
            msg = "LOGIT SHAP requires values strictly inside (0, 1)"
            raise ProviderInvalidResponseError(msg)
        return selected

    return predict


def prediction_callable(
    raw_callable: Callable[[np.ndarray], object],
    task: TaskType,
    *,
    allow_single_probability: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Adapt raw provider or estimator output to an explainer callable."""

    def predict(values: np.ndarray) -> np.ndarray:
        return normalize_predictions(
            raw_callable(values),
            task,
            len(values),
            allow_single_probability=allow_single_probability,
        )

    return predict
