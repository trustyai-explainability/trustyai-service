"""Normalize provider and sklearn outputs to algorithm callables."""

from collections.abc import Callable

import numpy as np

from .model_provider import ProviderInvalidResponseError
from .types import TaskType


def normalize_predictions(
    raw: object, task: TaskType, rows: int, *, allow_single_probability: bool = False
) -> np.ndarray:
    values = np.asarray(raw)
    if values.ndim == 0 or values.dtype.kind not in "biuf":
        raise ProviderInvalidResponseError("Predictions must be numeric arrays")
    if values.shape[0] != rows:
        raise ProviderInvalidResponseError("Prediction row count does not match input")
    if task is TaskType.REGRESSION:
        if values.ndim == 1:
            result = _finite_float(values, "Regression predictions")
            if not np.isfinite(result).all():
                raise ProviderInvalidResponseError(
                    "Regression predictions must be finite"
                )
            return result
        if values.ndim == 2 and values.shape[1] == 1:
            result = _finite_float(values[:, 0], "Regression predictions")
            if not np.isfinite(result).all():
                raise ProviderInvalidResponseError(
                    "Regression predictions must be finite"
                )
            return result
        raise ProviderInvalidResponseError(
            "Regression model must return one numeric output"
        )
    if values.ndim == 1:
        probability = _finite_float(values, "Classification probabilities")
        values = np.column_stack((1.0 - probability, probability))
    elif values.ndim == 2 and (
        values.shape[1] >= 2 or (values.shape[1] == 1 and allow_single_probability)
    ):
        values = _finite_float(values, "Classification probabilities")
    else:
        raise ProviderInvalidResponseError(
            "Classification model must return probabilities"
        )
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ProviderInvalidResponseError("Classification probabilities are invalid")
    if values.shape[1] > 1 and not np.allclose(
        values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-3
    ):
        raise ProviderInvalidResponseError(
            "Classification probabilities must sum to one"
        )
    return values


def _finite_float(values: np.ndarray, label: str) -> np.ndarray:
    try:
        result = values.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderInvalidResponseError(f"{label} must be numeric") from exc
    if not np.isfinite(result).all():
        raise ProviderInvalidResponseError(f"{label} must be finite")
    return result


def selected_class_callable(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    class_index: int,
    *,
    link: str = "IDENTITY",
    single_probability: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    def predict(values: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(predict_fn(values))
        if probabilities.ndim != 2:
            raise ProviderInvalidResponseError("Classification output must be a matrix")
        if probabilities.shape[1] == 1:
            if not single_probability or class_index != 1:
                raise ProviderInvalidResponseError("Selected class index is invalid")
            selected = probabilities[:, 0]
        elif not 0 <= class_index < probabilities.shape[1]:
            raise ProviderInvalidResponseError("Selected class index is invalid")
        else:
            selected = probabilities[:, class_index]
        selected = _finite_float(selected, "Selected class probabilities")
        if link.upper() == "LOGIT" and not np.all((selected > 0) & (selected < 1)):
            raise ProviderInvalidResponseError(
                "LOGIT SHAP requires probabilities strictly inside (0, 1)"
            )
        return selected

    return predict


def selected_scalar_callable(
    predict_fn: Callable[[np.ndarray], np.ndarray], *, link: str = "IDENTITY"
) -> Callable[[np.ndarray], np.ndarray]:
    def predict(values: np.ndarray) -> np.ndarray:
        raw = np.asarray(predict_fn(values))
        if raw.ndim == 2 and raw.shape[1] == 1:
            raw = raw[:, 0]
        if raw.ndim != 1 or raw.shape[0] != len(values):
            raise ProviderInvalidResponseError("Scalar model output is malformed")
        selected = _finite_float(raw, "Scalar model output")
        if link.upper() == "LOGIT" and not np.all((selected > 0) & (selected < 1)):
            raise ProviderInvalidResponseError(
                "LOGIT SHAP requires values strictly inside (0, 1)"
            )
        return selected

    return predict


def prediction_callable(
    raw_callable: Callable[[np.ndarray], object],
    task: TaskType,
    *,
    allow_single_probability: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    def predict(values: np.ndarray) -> np.ndarray:
        return normalize_predictions(
            raw_callable(values),
            task,
            len(values),
            allow_single_probability=allow_single_probability,
        )

    return predict
