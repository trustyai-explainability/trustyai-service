"""Tests for the POST /explainers/local/shap endpoint.

Request-model validation tests (``TestSHAPConfigValidation``) are kept as direct
tests. HTTP behaviour tests use ``tests.endpoints.explainers.local.factory``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
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

_MODULE = "trustyai_service.endpoints.explainers.local_explainer"
_MODEL_DATA_MODULE = "trustyai_service.service.data.model_data"

# ---------------------------------------------------------------------------
# Test client
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router)
client = TestClient(app)

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

    def test_n_samples_below_one_raises(self) -> None:
        """n_samples < 1 fails validation."""
        with pytest.raises(ValidationError, match="n_samples must be >= 1"):
            SHAPExplainerConfig(n_samples=0)

    def test_n_samples_above_max_raises(self) -> None:
        """n_samples > 50000 fails validation."""
        with pytest.raises(ValidationError, match="n_samples must be <= 50000"):
            SHAPExplainerConfig(n_samples=50_001)

    def test_n_samples_at_max_is_valid(self) -> None:
        """n_samples = 50000 is valid."""
        assert SHAPExplainerConfig(n_samples=50_000).n_samples == 50_000

    def test_n_training_rows_below_one_raises(self) -> None:
        """n_training_rows < 1 fails validation."""
        with pytest.raises(ValidationError, match="n_training_rows must be >= 1"):
            SHAPExplainerConfig(n_training_rows=0)

    def test_n_training_rows_above_max_raises(self) -> None:
        """n_training_rows > 1_000_000 fails validation."""
        with pytest.raises(ValidationError, match="n_training_rows must be <= 1000000"):
            SHAPExplainerConfig(n_training_rows=1_000_001)

    def test_timeout_below_one_raises(self) -> None:
        """Timeout < 1 fails validation."""
        with pytest.raises(ValidationError, match="timeout must be >= 1 second"):
            SHAPExplainerConfig(timeout=0)

    def test_timeout_above_max_raises(self) -> None:
        """Timeout > 3600 fails validation."""
        with pytest.raises(ValidationError, match="timeout must be <= 3600 seconds"):
            SHAPExplainerConfig(timeout=3601)


class TestSHAPLogitValidation:
    """Handler rejects LOGIT link when regressor outputs are out of [0, 1]."""

    @patch(f"{_MODULE}._SHAP_AVAILABLE", new=True)
    @patch(f"{_MODEL_DATA_MODULE}.get_global_storage_interface")
    @patch(f"{_MODULE}.get_shared_data_source")
    @patch(f"{_MODULE}.storage_interface")
    def test_logit_with_regressor_out_of_range_returns_400(
        self,
        mock_si: MagicMock,
        mock_ds_factory: MagicMock,
        mock_global: MagicMock,
    ) -> None:
        """LOGIT link with float (regression) outputs outside [0, 1] returns 400."""
        pred_id = "pred-logit-001"
        rng = np.random.default_rng(0)
        n_features = 3
        feature_names = [f"x{i}" for i in range(n_features)]
        metadata = np.array([[pred_id, "2025-01-01T00:00:00", 1.0, []]], dtype="O")
        input_row = rng.standard_normal((1, n_features)).astype(np.float64)
        organic_data = {name: rng.standard_normal(30) for name in feature_names}
        organic_data["output"] = rng.standard_normal(30) * 10.0  # outside [0, 1]
        organic_df = pd.DataFrame(organic_data)

        mock_si.dataset_exists = AsyncMock(return_value=True)
        mock_si.read_data = AsyncMock(
            side_effect=lambda name, *_a, **_kw: (
                metadata if "metadata" in name else input_row
            )
        )
        mock_si.get_aliased_column_names = AsyncMock(
            side_effect=lambda name: feature_names if "input" in name else ["output"]
        )
        mock_global.return_value = mock_si
        mock_ds = MagicMock()
        mock_ds.get_organic_dataframe = AsyncMock(return_value=organic_df)
        mock_ds_factory.return_value = mock_ds

        payload = {
            "predictionId": pred_id,
            "config": {
                "model": {"target": "regressor", "name": "test-model"},
                "explainer": {"link": "LOGIT", "confidence": 1.0, "n_samples": 10},
            },
        }
        response = client.post(_ENDPOINT, json=payload)
        assert response.status_code == 400
        assert "LOGIT" in response.json()["detail"]
        assert "[0, 1]" in response.json()["detail"]
