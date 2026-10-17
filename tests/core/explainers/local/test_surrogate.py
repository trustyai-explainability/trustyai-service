"""Tests for explicit local surrogate construction."""

import numpy as np
import pytest

from trustyai_service.core.explainers.local.surrogate import build_surrogate


def test_surrogate_mode_is_explicit() -> None:
    """Build the requested regression and classification surrogate models."""
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]])
    regression = build_surrogate(inputs, np.array([0.0, 1.0, 2.0, 3.0]), "REGRESSION")
    classification = build_surrogate(inputs, np.array([0, 1, 0, 1]), "CLASSIFICATION")
    assert regression.predict([[1.5]]).shape == (1,)
    assert classification.predict_proba([[1.5]]).shape == (1, 2)


def test_surrogate_rejects_unknown_mode() -> None:
    """Reject a surrogate mode that is not part of the public contract."""
    with pytest.raises(ValueError, match="CLASSIFICATION or REGRESSION"):
        build_surrogate(np.ones((2, 1)), np.ones(2), "UNKNOWN")  # type: ignore[arg-type]
