"""Tests for LIME endpoint models and validation.

Full endpoint integration testing requires fixture setup with storage and data.
Core computation tests cover LIME logic; these tests focus on request/response
models and validation.
"""

import pytest

from trustyai_service.endpoints.explainers.local_explainer import (
    LimeExplainerConfig,
    LimeExplanationRequest,
    LIMEExplanationResponse,
    LIMEFeatureAttribution,
)


class TestLimeExplainerConfig:
    """Tests for LIME explainer config models (independent of lime package)."""

    def test_lime_config_defaults(self) -> None:
        """LimeExplainerConfig has sensible defaults."""
        config = LimeExplainerConfig()

        assert config.num_samples == 5000
        assert config.n_training_rows == 1000
        assert config.kernel_width == 0.75
        assert config.num_features == 10
        assert config.timeout == 300
        assert config.confidence == 0.95

    def test_lime_config_confidence_validation(self) -> None:
        """LimeExplainerConfig validates confidence in (0, 1]."""
        # Valid: within (0, 1]
        config = LimeExplainerConfig(confidence=0.95)
        assert config.confidence == 0.95

        config = LimeExplainerConfig(confidence=0.5)
        assert config.confidence == 0.5

        config = LimeExplainerConfig(confidence=1.0)
        assert config.confidence == 1.0

        # Invalid: outside (0, 1]
        with pytest.raises(ValueError, match="confidence must be in"):
            LimeExplainerConfig(confidence=0.0)

        with pytest.raises(ValueError, match="confidence must be in"):
            LimeExplainerConfig(confidence=1.5)

    def test_num_samples_below_one_raises(self) -> None:
        """num_samples < 1 fails validation."""
        with pytest.raises(ValueError, match="num_samples must be >= 1"):
            LimeExplainerConfig(num_samples=0)

    def test_num_samples_above_max_raises(self) -> None:
        """num_samples > 100000 fails validation."""
        with pytest.raises(ValueError, match="num_samples must be <= 100000"):
            LimeExplainerConfig(num_samples=100_001)

    def test_n_training_rows_below_one_raises(self) -> None:
        """n_training_rows < 1 fails validation."""
        with pytest.raises(ValueError, match="n_training_rows must be >= 1"):
            LimeExplainerConfig(n_training_rows=0)

    def test_n_training_rows_above_max_raises(self) -> None:
        """n_training_rows > 1_000_000 fails validation."""
        with pytest.raises(ValueError, match="n_training_rows must be <= 1000000"):
            LimeExplainerConfig(n_training_rows=1_000_001)

    def test_kernel_width_zero_raises(self) -> None:
        """kernel_width <= 0 fails validation."""
        with pytest.raises(ValueError, match="kernel_width must be > 0"):
            LimeExplainerConfig(kernel_width=0.0)

    def test_kernel_width_above_max_raises(self) -> None:
        """kernel_width > 10.0 fails validation."""
        with pytest.raises(ValueError, match="kernel_width must be <= 10"):
            LimeExplainerConfig(kernel_width=10.1)

    def test_num_features_below_one_raises(self) -> None:
        """num_features < 1 fails validation."""
        with pytest.raises(ValueError, match="num_features must be >= 1"):
            LimeExplainerConfig(num_features=0)

    def test_num_features_above_max_raises(self) -> None:
        """num_features > 1000 fails validation."""
        with pytest.raises(ValueError, match="num_features must be <= 1000"):
            LimeExplainerConfig(num_features=1001)

    def test_timeout_below_one_raises(self) -> None:
        """Timeout < 1 fails validation."""
        with pytest.raises(ValueError, match="timeout must be >= 1 second"):
            LimeExplainerConfig(timeout=0)

    def test_timeout_above_max_raises(self) -> None:
        """Timeout > 3600 fails validation."""
        with pytest.raises(ValueError, match="timeout must be <= 3600 seconds"):
            LimeExplainerConfig(timeout=3601)


class TestLimeExplanationRequest:
    """Tests for LIME explanation request/response (requires lime package)."""

    def test_lime_request_parsing(self) -> None:
        """LimeExplanationRequest parses valid payloads."""
        payload = {
            "predictionId": "pred-123",
            "config": {
                "model": {
                    "target": "classifier",
                    "name": "test_model",
                },
                "explainer": {
                    "num_samples": 5000,
                    "confidence": 0.95,
                },
            },
        }

        request = LimeExplanationRequest.model_validate(payload)

        assert request.predictionId == "pred-123"
        assert request.config.model.name == "test_model"
        assert request.config.explainer is not None
        assert request.config.explainer.confidence == 0.95

    def test_lime_request_with_none_explainer(self) -> None:
        """LimeExplanationRequest allows null explainer (uses defaults)."""
        payload = {
            "predictionId": "pred-456",
            "config": {
                "model": {
                    "target": "regressor",
                    "name": "another_model",
                },
                "explainer": None,
            },
        }

        request = LimeExplanationRequest.model_validate(payload)

        assert request.predictionId == "pred-456"
        assert request.config.explainer is None


class TestLIMEExplanationResponse:
    """Bug 2: Validate LIMEExplanationResponse field names match handler output."""

    def test_response_field_names_exist(self) -> None:
        """LIMEExplanationResponse has all required fields with correct names."""
        # Verify expected field names exist in the response model
        response = LIMEExplanationResponse(
            prediction_id="pred-123",
            model="test_model",
            attributions=[
                LIMEFeatureAttribution(
                    feature_name="feature_0",
                    importance=0.5,
                    confidence_lower=None,
                    confidence_upper=None,
                )
            ],
            score=0.95,
            local_prediction=0.42,
            intercept=0.1,
        )

        # Verify all fields exist with correct names (Bug 2)
        assert response.prediction_id == "pred-123"
        assert response.model == "test_model"
        assert response.score == 0.95
        assert response.local_prediction == 0.42  # NOT "local_pred"
        assert response.intercept == 0.1
        assert len(response.attributions) == 1

    def test_response_json_serialization(self) -> None:
        """LIMEExplanationResponse serializes to JSON with correct field names."""
        response = LIMEExplanationResponse(
            prediction_id="pred-123",
            model="test_model",
            attributions=[],
            score=0.95,
            local_prediction=0.42,
            intercept=0.1,
        )

        json_data = response.model_dump()
        # Verify JSON keys match expected names
        assert "prediction_id" in json_data
        assert "model" in json_data
        assert "score" in json_data
        assert "local_prediction" in json_data
        assert "intercept" in json_data
        assert "attributions" in json_data
