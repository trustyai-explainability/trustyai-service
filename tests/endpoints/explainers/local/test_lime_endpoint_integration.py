"""Integration tests for LIME endpoint using the local explainer factory.

Tests the full request/response cycle with mocked storage and computation,
validating:
- Correct tuple shape for _compute() return (Bug 1)
- Response model field names match handler output (Bug 2)
- Proper async mock setup (Bug 3)
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.endpoints import routes
from trustyai_service.endpoints.explainers.local_explainer import router

from . import factory

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


class TestLimeEndpointIntegration:
    """Integration tests for LIME endpoint using mocked storage and computation."""

    def test_lime_endpoint_happy_path_with_ci(self) -> None:
        """Bug 1 & 2: Verify LIME endpoint returns correct response shape.

        Tests that:
        - The mocked _compute() returns a 6-tuple with correct ordering:
          (weights, r2, local_pred, intercept, lower, upper)
        - Response fields match LIMEExplanationResponse model:
          prediction_id, model, attributions, score, local_prediction, intercept
        """
        feature_names = ["feature_0", "feature_1", "feature_2"]
        output_names = ["output"]
        pred_id = "pred-123"

        # Create fake _compute() return with correct 6-tuple ordering:
        # weights, r2, local_pred, intercept, lower, upper
        fake_weights = [
            ("feature_0", 0.5),
            ("feature_1", 0.3),
        ]
        fake_r2 = 0.95
        fake_local_pred = 0.42
        fake_intercept = 0.1
        fake_lower = {"feature_0": 0.4, "feature_1": 0.2}
        fake_upper = {"feature_0": 0.6, "feature_1": 0.4}

        # Correct tuple order matching handler line ~440
        fake_compute_return = (
            fake_weights,
            fake_r2,
            fake_local_pred,
            fake_intercept,
            fake_lower,
            fake_upper,
        )

        request_payload = {
            "predictionId": pred_id,
            "config": {
                "model": {
                    "target": "regressor",
                    "name": "test_model",
                },
                "explainer": {
                    "num_samples": 1000,
                    "confidence": 0.95,
                },
            },
        }

        # Use factory to create the test with the correct tuple return
        test_func = factory.make_compute_endpoint_test(
            explainer_name="LIME",
            endpoint_path=routes.EXPLAINER_LOCAL_LIME,
            client=client,
            request_payload=request_payload,
            # Bug 2: Verify these response keys match LIMEExplanationResponse fields
            expected_response_keys=[
                "prediction_id",
                "model",
                "attributions",
                "score",
                "local_prediction",  # NOT "local_pred"
                "intercept",
            ],
            feature_names=feature_names,
            compute_thread_return=fake_compute_return,
            availability_flag="trustyai_service.endpoints.explainers.local_explainer._LIME_AVAILABLE",
            output_names=output_names,
            pred_id=pred_id,
        )

        test_func(self)

    def test_lime_endpoint_happy_path_without_ci(self) -> None:
        """LIME endpoint with confidence=1.0 (skips CI computation)."""
        feature_names = ["feature_0", "feature_1"]
        output_names = ["output"]
        pred_id = "pred-456"

        # With confidence=1.0, CI is disabled, so lower and upper are None
        fake_compute_return = (
            [("feature_0", 0.5)],
            0.9,
            0.5,
            0.2,
            None,  # lower is None
            None,  # upper is None
        )

        request_payload = {
            "predictionId": pred_id,
            "config": {
                "model": {
                    "target": "regressor",
                    "name": "another_model",
                },
                "explainer": {
                    "num_samples": 5000,
                    "confidence": 1.0,  # CI disabled
                },
            },
        }

        test_func = factory.make_compute_endpoint_test(
            explainer_name="LIME",
            endpoint_path=routes.EXPLAINER_LOCAL_LIME,
            client=client,
            request_payload=request_payload,
            expected_response_keys=[
                "prediction_id",
                "model",
                "attributions",
                "score",
                "local_prediction",
                "intercept",
            ],
            feature_names=feature_names,
            compute_thread_return=fake_compute_return,
            availability_flag="trustyai_service.endpoints.explainers.local_explainer._LIME_AVAILABLE",
            output_names=output_names,
            pred_id=pred_id,
        )

        test_func(self)

    def test_lime_unavailable_503(self) -> None:
        """503 SERVICE_UNAVAILABLE when lime package not installed."""
        request_payload = {
            "predictionId": "pred-999",
            "config": {
                "model": {
                    "target": "classifier",
                    "name": "test_model",
                },
            },
        }

        test_func = factory.make_explainer_unavailable_test(
            explainer_name="LIME",
            availability_flag="trustyai_service.endpoints.explainers.local_explainer._LIME_AVAILABLE",
            endpoint_path=routes.EXPLAINER_LOCAL_LIME,
            client=client,
            request_payload=request_payload,
        )

        test_func(self)

    def test_lime_model_not_found_404(self) -> None:
        """404 NOT_FOUND when model has no stored data."""
        request_payload = {
            "predictionId": "pred-999",
            "config": {
                "model": {
                    "target": "classifier",
                    "name": "unknown_model",
                },
            },
        }

        test_func = factory.make_model_not_found_test(
            explainer_name="LIME",
            endpoint_path=routes.EXPLAINER_LOCAL_LIME,
            client=client,
            request_payload=request_payload,
            availability_flag="trustyai_service.endpoints.explainers.local_explainer._LIME_AVAILABLE",
        )

        test_func(self)

    def test_lime_prediction_id_not_found_400(self) -> None:
        """400 BAD_REQUEST when predictionId not in stored metadata."""
        request_payload = {
            "predictionId": "nonexistent-pred",  # Does not match stored pred-123
            "config": {
                "model": {
                    "target": "classifier",
                    "name": "test_model",
                },
            },
        }

        test_func = factory.make_prediction_id_not_found_test(
            explainer_name="LIME",
            endpoint_path=routes.EXPLAINER_LOCAL_LIME,
            client=client,
            request_payload=request_payload,
            availability_flag="trustyai_service.endpoints.explainers.local_explainer._LIME_AVAILABLE",
        )

        test_func(self)

    def test_lime_missing_prediction_id_422(self) -> None:
        """422 UNPROCESSABLE_ENTITY when predictionId field is missing."""
        invalid_payload = {
            # Missing "predictionId"
            "config": {
                "model": {
                    "target": "classifier",
                    "name": "test_model",
                },
            },
        }

        test_func = factory.make_missing_field_validation_test(
            explainer_name="LIME",
            endpoint_path=routes.EXPLAINER_LOCAL_LIME,
            client=client,
            invalid_payload=invalid_payload,
            expected_field="predictionId",
        )

        test_func(self)
