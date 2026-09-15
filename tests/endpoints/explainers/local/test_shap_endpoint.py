"""Tests for the POST /explainers/local/shap endpoint.

Request-model validation tests (``TestSHAPConfigValidation``) are kept as direct
tests. HTTP behaviour tests use ``tests.endpoints.explainers.local.factory``.
"""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.endpoints.explainers.local import factory
from trustyai_service.endpoints import routes
from trustyai_service.endpoints.explainers.local_explainer import (
    SHAPExplainerConfig,
    router,
)

# ---------------------------------------------------------------------------
# Test client
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router)
client = TestClient(app)

_MODULE = "trustyai_service.endpoints.explainers.local_explainer"
_ENDPOINT = routes.EXPLAINER_LOCAL_SHAP
_AVAILABILITY_FLAG = f"{_MODULE}._SHAP_AVAILABLE"
_FEATURE_NAMES = ["x0", "x1", "x2"]
_PRED_ID = "pred-shap-001"

_VALID_PAYLOAD: dict = {
    "predictionId": _PRED_ID,
    "config": {
        "model": {"target": "sklearn", "name": "test-model"},
        "explainer": {
            "n_samples": 10,
            "confidence": 1.0,
            "link": "IDENTITY",
            "regularizer": "AUTO",
        },
    },
}

# Handler unpacks: shap_vals, lower, upper = await asyncio.wait_for(asyncio.to_thread(_compute), ...)
# _compute() returns tuple[np.ndarray, np.ndarray | None, np.ndarray | None].
# confidence=1.0 disables CI so lower=None, upper=None.
# shap_vals must have one element per feature (3 features: x0, x1, x2).
_FAKE_SHAP_COMPUTE_RETURN = (np.array([0.5, -0.1, 0.2]), None, None)


class TestSHAPEndpointContract:
    """HTTP contract coverage for /explainers/local/shap."""

    test_returns_200_with_expected_keys = factory.make_compute_endpoint_test(
        explainer_name="SHAP",
        endpoint_path=_ENDPOINT,
        client=client,
        request_payload=_VALID_PAYLOAD,
        expected_response_keys=[
            "prediction_id",
            "model",
            "attributions",
            "link",
            "regularizer",
        ],
        feature_names=_FEATURE_NAMES,
        compute_thread_return=_FAKE_SHAP_COMPUTE_RETURN,
        availability_flag=_AVAILABILITY_FLAG,
        pred_id=_PRED_ID,
    )

    test_unavailable_returns_503 = factory.make_explainer_unavailable_test(
        explainer_name="SHAP",
        availability_flag=_AVAILABILITY_FLAG,
        endpoint_path=_ENDPOINT,
        client=client,
        request_payload=_VALID_PAYLOAD,
    )

    test_model_not_found_returns_404 = factory.make_model_not_found_test(
        explainer_name="SHAP",
        endpoint_path=_ENDPOINT,
        client=client,
        request_payload=_VALID_PAYLOAD,
        availability_flag=_AVAILABILITY_FLAG,
    )

    test_unknown_pred_id_returns_400 = factory.make_prediction_id_not_found_test(
        explainer_name="SHAP",
        endpoint_path=_ENDPOINT,
        client=client,
        request_payload=_VALID_PAYLOAD,
        availability_flag=_AVAILABILITY_FLAG,
    )

    test_missing_pred_id_returns_422 = factory.make_missing_field_validation_test(
        explainer_name="SHAP",
        endpoint_path=_ENDPOINT,
        client=client,
        invalid_payload={
            "config": {
                "model": {"target": "sklearn", "name": "test-model"},
            }
        },
        expected_field="predictionId",
    )


class TestSHAPConfigValidation:
    """Pydantic validation for SHAPExplainerConfig."""

    def test_confidence_below_zero_raises(self) -> None:
        """Confidence < 0 fails validation."""
        with pytest.raises(ValidationError, match="confidence must be in"):
            SHAPExplainerConfig(confidence=-0.1)

    def test_confidence_above_one_raises(self) -> None:
        """Confidence > 1 fails validation."""
        with pytest.raises(ValidationError, match="confidence must be in"):
            SHAPExplainerConfig(confidence=1.5)

    def test_confidence_one_is_valid(self) -> None:
        """Confidence = 1.0 (sentinel) is valid."""
        assert SHAPExplainerConfig(confidence=1.0).confidence == 1.0

    def test_confidence_just_below_one_is_valid(self) -> None:
        """Confidence < 1.0 is valid."""
        assert SHAPExplainerConfig(confidence=0.99).confidence == 0.99
