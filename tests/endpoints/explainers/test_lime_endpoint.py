"""Tests for LIME endpoint models and validation.

Full endpoint integration testing requires fixture setup with storage and data.
Core computation tests cover LIME logic; these tests focus on request/response
models and validation.
"""

import pytest

from trustyai_service.core.explainers.lime import _LIME_AVAILABLE
from trustyai_service.endpoints.explainers.local_explainer import (
    LimeExplainerConfig,
    LimeExplanationRequest,
)


@pytest.mark.skipif(not _LIME_AVAILABLE, reason="lime not installed")
class TestLimeEndpointModels:
    """Tests for LIME endpoint request/response models."""

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
