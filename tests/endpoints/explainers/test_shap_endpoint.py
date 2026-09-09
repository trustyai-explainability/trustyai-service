"""Integration tests for the POST /explainers/local/shap endpoint.

Uses the FastAPI TestClient with mocked storage.  The shap package must be
installed (skip otherwise).
"""

import importlib
from http import HTTPStatus
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

pytest.importorskip("shap", reason="shap extra not installed")

from trustyai_service.endpoints import routes
from trustyai_service.endpoints.explainers.local_explainer import SHAPExplainerConfig
from trustyai_service.service.config import feature_flags

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES = 3
N_ROWS = 30
FEATURE_NAMES = ["x0", "x1", "x2"]
MODEL_NAME = "test-model"
PRED_ID = "pred-abc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata_array(pred_ids: list[str]) -> np.ndarray:
    """Build a metadata array with id in column 0."""
    rows = [[pid, "2025-01-01T00:00:00", 1.0, []] for pid in pred_ids]
    return np.array(rows, dtype="O")


def _make_input_row() -> np.ndarray:
    """Single input row of shape (1, N_FEATURES)."""
    return np.array([[1.0, 0.5, -0.3]], dtype=np.float64)


def _make_organic_df() -> pd.DataFrame:
    """Organic dataframe with input + output columns."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_ROWS, N_FEATURES))
    y = X[:, 0] * 2.0
    data: dict[str, np.ndarray] = {
        name: X[:, i] for i, name in enumerate(FEATURE_NAMES)
    }
    data["output"] = y
    return pd.DataFrame(data)


def _flags_with_explainers() -> dict[str, bool]:
    return {**feature_flags.ENDPOINTS, "explainer": True, "explainer_local": True}


def _valid_request_body() -> dict:
    return {
        "predictionId": PRED_ID,
        "config": {
            "model": {"target": "sklearn", "name": MODEL_NAME},
            "explainer": {
                "n_samples": 10,
                "confidence": 1.0,
                "link": "IDENTITY",
                "regularizer": "AUTO",
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSHAPEndpointValid:
    """Happy-path tests for /explainers/local/shap."""

    @patch("trustyai_service.endpoints.explainers.local_explainer.storage_interface")
    @patch("trustyai_service.service.data.model_data.get_global_storage_interface")
    @patch(
        "trustyai_service.endpoints.explainers.local_explainer.get_shared_data_source"
    )
    def test_valid_request_returns_200_with_attributions(
        self,
        mock_ds_factory: AsyncMock,
        mock_global: AsyncMock,
        mock_si: AsyncMock,
    ) -> None:
        """Valid request returns 200 with one attribution per feature."""
        from fastapi.testclient import TestClient  # noqa: PLC0415

        metadata = _make_metadata_array([PRED_ID])
        input_row = _make_input_row()
        organic = _make_organic_df()

        mock_si.dataset_exists = AsyncMock(return_value=True)
        mock_si.read_data = AsyncMock(
            side_effect=lambda name, *_args, **_kwargs: (
                metadata if "metadata" in name else input_row
            )
        )
        mock_si.get_aliased_column_names = AsyncMock(
            side_effect=lambda name: FEATURE_NAMES if "input" in name else ["output"]
        )
        mock_global.return_value = mock_si

        mock_ds = AsyncMock()
        mock_ds.get_organic_dataframe = AsyncMock(return_value=organic)
        mock_ds_factory.return_value = mock_ds

        with patch.dict(feature_flags.ENDPOINTS, _flags_with_explainers()):
            importlib.reload(importlib.import_module("trustyai_service.main"))
            from trustyai_service import main as _main  # noqa: PLC0415

            importlib.reload(_main)
            with (
                patch(
                    "trustyai_service.endpoints.explainers.local_explainer.storage_interface",
                    mock_si,
                ),
                TestClient(_main.app) as client,
            ):
                response = client.post(
                    routes.EXPLAINER_LOCAL_SHAP, json=_valid_request_body()
                )

        assert response.status_code == HTTPStatus.OK
        body = response.json()
        assert "attributions" in body
        assert len(body["attributions"]) == N_FEATURES
        for attr in body["attributions"]:
            assert "feature_name" in attr
            assert "shap_value" in attr


class TestSHAPEndpointErrors:
    """Error-path tests for /explainers/local/shap."""

    def test_invalid_link_value_returns_422(self) -> None:
        """Unrecognised link enum value returns 422 Unprocessable Entity."""
        from fastapi.testclient import TestClient  # noqa: PLC0415

        body = _valid_request_body()
        body["config"]["explainer"]["link"] = "INVALID_LINK"  # type: ignore[index]

        with patch.dict(feature_flags.ENDPOINTS, _flags_with_explainers()):
            importlib.reload(importlib.import_module("trustyai_service.main"))
            from trustyai_service import main as _main  # noqa: PLC0415

            importlib.reload(_main)
            with TestClient(_main.app) as client:
                response = client.post(routes.EXPLAINER_LOCAL_SHAP, json=body)

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    @patch("trustyai_service.endpoints.explainers.local_explainer.storage_interface")
    def test_unknown_prediction_id_returns_400(self, mock_si: AsyncMock) -> None:
        """Unknown predictionId returns 400 Bad Request."""
        from fastapi.testclient import TestClient  # noqa: PLC0415

        mock_si.dataset_exists = AsyncMock(return_value=True)
        mock_si.read_data = AsyncMock(return_value=_make_metadata_array(["other-pred"]))

        with (
            patch.dict(feature_flags.ENDPOINTS, _flags_with_explainers()),
            patch(
                "trustyai_service.endpoints.explainers.local_explainer.storage_interface",
                mock_si,
            ),
            patch(
                "trustyai_service.service.data.model_data.get_global_storage_interface",
                return_value=mock_si,
            ),
        ):
            importlib.reload(importlib.import_module("trustyai_service.main"))
            from trustyai_service import main as _main  # noqa: PLC0415

            importlib.reload(_main)
            with TestClient(_main.app) as client:
                response = client.post(
                    routes.EXPLAINER_LOCAL_SHAP, json=_valid_request_body()
                )

        assert response.status_code == HTTPStatus.BAD_REQUEST

    @patch(
        "trustyai_service.endpoints.explainers.local_explainer._SHAP_AVAILABLE",
        False,
    )
    def test_shap_unavailable_returns_503(self) -> None:
        """When SHAP is not installed the endpoint returns 503."""
        from fastapi.testclient import TestClient  # noqa: PLC0415

        with patch.dict(feature_flags.ENDPOINTS, _flags_with_explainers()):
            importlib.reload(importlib.import_module("trustyai_service.main"))
            from trustyai_service import main as _main  # noqa: PLC0415

            importlib.reload(_main)
            with TestClient(_main.app) as client:
                response = client.post(
                    routes.EXPLAINER_LOCAL_SHAP, json=_valid_request_body()
                )

        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE


# ---------------------------------------------------------------------------
# Configuration validation tests
# ---------------------------------------------------------------------------


class TestSHAPConfigValidation:
    """Test Pydantic validation of SHAPExplainerConfig."""

    def test_confidence_below_zero_raises_validation_error(self) -> None:
        """Confidence < 0 fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            SHAPExplainerConfig(confidence=-0.1)

        assert "confidence must be in (0, 1]" in str(exc_info.value)

    def test_confidence_above_one_raises_validation_error(self) -> None:
        """Confidence > 1 fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            SHAPExplainerConfig(confidence=1.5)

        assert "confidence must be in (0, 1]" in str(exc_info.value)

    def test_confidence_exactly_one_is_valid(self) -> None:
        """Confidence = 1.0 (sentinel) is valid."""
        config = SHAPExplainerConfig(confidence=1.0)
        assert config.confidence == 1.0

    def test_confidence_just_below_one_is_valid(self) -> None:
        """Confidence < 1.0 is valid."""
        config = SHAPExplainerConfig(confidence=0.99)
        assert config.confidence == 0.99
