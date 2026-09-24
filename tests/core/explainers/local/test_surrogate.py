"""Tests for the explicit local random-forest surrogate boundary."""

import numpy as np
import pytest

from trustyai_service.core.explainers.local.surrogate import build_surrogate
from trustyai_service.service.explainers.local.types import TaskType


def test_build_surrogate_supports_explicit_regression_and_classification() -> None:
    """Fit the requested RF task and expose its native prediction shape."""
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]])

    regression = build_surrogate(
        inputs,
        np.array([0.0, 1.0, 2.0, 3.0]),
        TaskType.REGRESSION,
    )
    classification = build_surrogate(
        inputs,
        np.array([0, 1, 0, 1]),
        TaskType.CLASSIFICATION,
    )

    assert regression.predict([[1.5]]).shape == (1,)
    assert classification.predict_proba([[1.5]]).shape == (1, 2)


def test_build_surrogate_rejects_unknown_task() -> None:
    """Do not silently select a surrogate for an unsupported task value."""
    with pytest.raises(ValueError, match="CLASSIFICATION or REGRESSION"):
        build_surrogate(
            np.ones((2, 1)),
            np.ones(2),
            "UNKNOWN",  # type: ignore[arg-type]
        )
