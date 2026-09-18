"""Tests for callable-based KernelSHAP result semantics."""

import numpy as np
import pytest

from trustyai_service.core.explainers.local.shap import (
    _SHAP_AVAILABLE,
    compute_shap_result,
)


@pytest.mark.skipif(not _SHAP_AVAILABLE, reason="SHAP extra is unavailable")
def test_shap_result_keeps_link_space_prediction_and_base_value() -> None:
    background = np.array([[0.1, 0.2], [0.2, 0.1]])
    instance = np.array([0.3, 0.4])

    def predict(values: np.ndarray) -> np.ndarray:
        return values.sum(axis=1) / 2.0

    result = compute_shap_result(
        instance,
        background,
        predict,
        n_samples=8,
        link="identity",
        l1_reg="auto",
    )
    assert result.values.shape == (2,)
    assert result.base_value + result.values.sum() == pytest.approx(
        result.linked_prediction, abs=1e-5
    )
