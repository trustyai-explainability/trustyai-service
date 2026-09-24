"""Normalize raw model and surrogate predictions for local explainers."""

from collections.abc import Callable
from numbers import Integral

import numpy as np

from .model_provider import (
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
)
from .types import TaskType

_MATRIX_RANK = 2
_MIN_CLASS_COUNT = 2
_SUPPORTED_LINKS = frozenset({"IDENTITY", "LOGIT"})
_PREDICTIONS_NOT_NUMERIC = "Predictions must be numeric arrays"
_PREDICTION_ROWS_INVALID = "Prediction row count does not match input"
_PREDICTIONS_NOT_FINITE = "Predictions must be finite"
_PREDICTION_ROW_COUNT_INVALID = "Prediction row count is invalid"
_REGRESSION_OUTPUT_INVALID = "Regression model must return one numeric output"
_CLASSIFICATION_OUTPUT_INVALID = "Classification model must return probabilities"
_CLASSIFICATION_PROBABILITIES_INVALID = "Classification probabilities are invalid"
_CLASSIFICATION_SUM_INVALID = "Classification probabilities must sum to one"
_TASK_INVALID = "Task must be CLASSIFICATION or REGRESSION"
_LINK_INVALID = "SHAP link must be IDENTITY or LOGIT"
_CLASS_INDEX_INVALID = "class_index is invalid"
_SINGLE_PROBABILITY_NOT_DECLARED = "A single probability requires explicit opt-in"
_WIDER_CLASS_INDEX_REQUIRED = "class_index is required for wider classification output"
_LOGIT_PROBABILITIES_INVALID = (
    "LOGIT SHAP requires probabilities strictly inside (0, 1)"
)
_LOGIT_VALUES_INVALID = "LOGIT SHAP requires values strictly inside (0, 1)"


def _invalid_response(message: str) -> ProviderInvalidResponseError:
    """Build a typed response error for malformed raw predictions."""
    return ProviderInvalidResponseError(message)


def _as_numeric_array(raw: object, rows: int) -> np.ndarray:
    """Convert raw output only when it is already a numeric finite array."""
    try:
        values = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise _invalid_response(_PREDICTIONS_NOT_NUMERIC) from exc

    if values.ndim == 0 or values.dtype.kind not in "biuf":
        raise _invalid_response(_PREDICTIONS_NOT_NUMERIC)
    if values.shape[0] != rows:
        raise _invalid_response(_PREDICTION_ROWS_INVALID)
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid_response(_PREDICTIONS_NOT_NUMERIC) from exc
    if not np.isfinite(numeric).all():
        raise _invalid_response(_PREDICTIONS_NOT_FINITE)
    return numeric


def _validate_rows(rows: int) -> None:
    """Require the row count supplied by the callable boundary to be integral."""
    if isinstance(rows, bool) or not isinstance(rows, Integral) or rows < 0:
        raise ProviderInvalidRequestError(_PREDICTION_ROW_COUNT_INVALID)


def _normalize_regression(values: np.ndarray) -> np.ndarray:
    """Normalize a scalar regression output to one value per row."""
    if values.ndim == 1:
        return values
    if values.ndim == _MATRIX_RANK and values.shape[1] == 1:
        return values[:, 0]
    raise _invalid_response(_REGRESSION_OUTPUT_INVALID)


def _normalize_classification(
    values: np.ndarray,
    *,
    allow_single_probability: bool,
) -> np.ndarray:
    """Validate a classification probability matrix for an explainer."""
    if values.ndim == 1:
        if allow_single_probability:
            values = values.reshape(-1, 1)
        else:
            raise _invalid_response(_CLASSIFICATION_OUTPUT_INVALID)
    if values.ndim != _MATRIX_RANK:
        raise _invalid_response(_CLASSIFICATION_OUTPUT_INVALID)
    if values.shape[1] < _MIN_CLASS_COUNT and not (
        values.shape[1] == 1 and allow_single_probability
    ):
        raise _invalid_response(_CLASSIFICATION_OUTPUT_INVALID)
    if values.shape[1] == 0:
        raise _invalid_response(_CLASSIFICATION_OUTPUT_INVALID)
    if np.any(values < 0) or np.any(values > 1):
        raise _invalid_response(_CLASSIFICATION_PROBABILITIES_INVALID)
    if values.shape[1] > 1 and not np.allclose(
        values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-3
    ):
        raise _invalid_response(_CLASSIFICATION_SUM_INVALID)
    return values


def normalize_predictions(
    raw: object,
    task: TaskType,
    rows: int,
    *,
    allow_single_probability: bool = False,
) -> np.ndarray:
    """Normalize raw provider or RF output to the requested task shape.

    Regression returns one value per row. Classification returns a probability
    matrix, requiring at least two columns unless the caller explicitly opts in
    to a one-column positive probability for SHAP.
    """
    _validate_rows(rows)
    values = _as_numeric_array(raw, int(rows))
    if task == TaskType.REGRESSION:
        return _normalize_regression(values)
    if task == TaskType.CLASSIFICATION:
        return _normalize_classification(
            values,
            allow_single_probability=allow_single_probability,
        )
    raise ProviderInvalidRequestError(_TASK_INVALID)


def prediction_callable(
    raw_callable: Callable[[np.ndarray], object],
    task: TaskType,
    *,
    allow_single_probability: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Adapt a raw provider or estimator callable to an explainer callable."""

    def predict(values: np.ndarray) -> np.ndarray:
        return normalize_predictions(
            raw_callable(values),
            task,
            len(values),
            allow_single_probability=allow_single_probability,
        )

    return predict


def _normalize_link(link: str) -> str:
    """Validate and normalize a SHAP link name without applying the link."""
    if not isinstance(link, str) or link.upper() not in _SUPPORTED_LINKS:
        raise ProviderInvalidRequestError(_LINK_INVALID)
    return link.upper()


def _validate_class_index(class_index: object, width: int) -> int:
    """Validate a selected class index against an output width."""
    if isinstance(class_index, bool) or not isinstance(class_index, Integral):
        raise ProviderInvalidRequestError(_CLASS_INDEX_INVALID)
    selected = int(class_index)
    if not 0 <= selected < width:
        raise ProviderInvalidRequestError(_CLASS_INDEX_INVALID)
    return selected


def selected_class_callable(
    predict_fn: Callable[[np.ndarray], object],
    class_index: int | None = None,
    *,
    link: str = "IDENTITY",
    single_probability: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Select one raw class probability for SHAP.

    Binary output defaults to class 1. Wider output requires an explicit
    ``class_index``. A one-column positive probability is accepted only when
    ``single_probability`` is true and represents class 1.
    """
    selected_link = _normalize_link(link)

    def predict(values: np.ndarray) -> np.ndarray:
        probabilities = normalize_predictions(
            predict_fn(values),
            TaskType.CLASSIFICATION,
            len(values),
            allow_single_probability=single_probability,
        )
        width = probabilities.shape[1]
        if width == 1:
            if not single_probability:
                raise _invalid_response(_SINGLE_PROBABILITY_NOT_DECLARED)
            selected = (
                1 if class_index is None else _validate_class_index(class_index, 2)
            )
            if selected != 1:
                raise ProviderInvalidRequestError(_CLASS_INDEX_INVALID)
            result = probabilities[:, 0]
        else:
            selected = 1 if class_index is None else class_index
            if selected is None:
                raise ProviderInvalidRequestError(_WIDER_CLASS_INDEX_REQUIRED)
            if class_index is None and width != _MIN_CLASS_COUNT:
                raise ProviderInvalidRequestError(_WIDER_CLASS_INDEX_REQUIRED)
            result = probabilities[:, _validate_class_index(selected, width)]

        if selected_link == "LOGIT" and not np.all((result > 0) & (result < 1)):
            raise _invalid_response(_LOGIT_PROBABILITIES_INVALID)
        return result

    return predict


def selected_scalar_callable(
    predict_fn: Callable[[np.ndarray], object],
    *,
    link: str = "IDENTITY",
) -> Callable[[np.ndarray], np.ndarray]:
    """Select a finite scalar output for SHAP without transforming it."""
    selected_link = _normalize_link(link)

    def predict(values: np.ndarray) -> np.ndarray:
        result = normalize_predictions(
            predict_fn(values),
            TaskType.REGRESSION,
            len(values),
        )
        if selected_link == "LOGIT" and not np.all((result > 0) & (result < 1)):
            raise _invalid_response(_LOGIT_VALUES_INVALID)
        return result

    return predict
