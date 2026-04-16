from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.drift.kolmogorov_smirnov_streaming import ApproxKSTestMetricRequest, router

from . import factory

# Create test app with KS Test Streaming router
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestKSTestStreamingEndpoints:
    """Unified endpoint tests for KS Test Streaming metric."""

    # Pandas DataFrame tests
    test_compute_endpoint_pandas = factory.make_compute_endpoint_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "epsilon": 0.01,
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha", "epsilon"],
        df_type="Pandas",
    )

    # Polars DataFrame tests
    test_compute_endpoint_polars = factory.make_compute_endpoint_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "epsilon": 0.01,
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha", "epsilon"],
        df_type="Polars",
    )

    test_definition_endpoint = factory.make_definition_endpoint_test(
        metric_name="KSTestStreaming",
        endpoint_path="/metrics/drift/ksteststreaming/definition",
        client=client,
        expected_name="Kolmogorov-Smirnov",
    )

    test_schedule_endpoint = factory.make_schedule_endpoint_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "epsilon": 0.01,
        },
    )

    test_delete_schedule_endpoint = factory.make_delete_schedule_endpoint_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
    )

    test_list_requests_endpoint = factory.make_list_requests_endpoint_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/requests",
        client=client,
    )

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    # Compute endpoint errors
    test_compute_missing_reference_tag = factory.make_compute_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming",
        client=client,
        request_payload={
            "modelId": "test-model",
            # Missing referenceTag
            "fitColumns": ["feature1"],
        },
        expected_status_code=400,
        expected_error_substring="referenceTag is required",
    )

    test_compute_missing_fit_columns = factory.make_compute_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            # Missing fitColumns
        },
        expected_status_code=400,
        expected_error_substring="fitColumns is required",
    )

    test_compute_invalid_feature = factory.make_compute_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["nonexistent_feature"],
        },
        expected_status_code=400,
        expected_error_substring="not found in data",
    )

    # Delete endpoint errors
    test_delete_invalid_uuid = factory.make_delete_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
        request_id="not-a-valid-uuid",
        expected_status_code=400,  # Now properly returns 400 for invalid UUID
        expected_error_substring="Invalid request ID",
    )

    # List endpoint with multiple requests
    test_list_requests_with_data = factory.make_list_requests_with_data_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/requests",
        client=client,
        num_requests=3,
    )

    # List endpoint with malformed requests (defensive logic test)
    test_list_requests_filters_malformed = factory.make_list_requests_with_malformed_data_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/requests",
        client=client,
        num_valid_requests=2,
        num_malformed_requests=3,
    )

    # Empty data tests
    test_compute_empty_reference_data = factory.make_compute_empty_reference_data_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    test_compute_empty_current_data = factory.make_compute_empty_current_data_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    # List endpoint error tests
    test_list_scheduler_unavailable = factory.make_list_endpoint_scheduler_unavailable_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/requests",
        client=client,
    )

    test_list_exception = factory.make_list_endpoint_exception_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/requests",
        client=client,
    )

    # Generic exception in compute endpoint
    test_compute_generic_exception = factory.make_compute_generic_exception_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    # ========================================================================
    # Scheduler Exception Tests
    # ========================================================================

    # Schedule endpoint scheduler exceptions
    test_schedule_connection_error = factory.make_schedule_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=500,
        expected_error_substring="connect",  # Match "Failed to connect"
        mock_scheduler_none=False,
        register_side_effect=ConnectionError("Failed to connect to scheduler database"),
    )

    test_schedule_timeout_error = factory.make_schedule_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=500,
        expected_error_substring="timeout",
        mock_scheduler_none=False,
        register_side_effect=TimeoutError("Scheduler registration timeout after 30s"),
    )

    test_schedule_runtime_error = factory.make_schedule_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=500,
        expected_error_substring="error",
        mock_scheduler_none=False,
        register_side_effect=RuntimeError("Internal scheduler error occurred"),
    )

    # Delete endpoint scheduler exceptions
    test_delete_connection_error = factory.make_delete_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,
        expected_error_substring="connect",  # Match "Failed to connect"
        mock_scheduler_none=False,
        delete_side_effect=ConnectionError("Failed to connect to scheduler database"),
    )

    test_delete_runtime_error = factory.make_delete_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,
        expected_error_substring="failed",
        mock_scheduler_none=False,
        delete_side_effect=RuntimeError("Scheduler deletion failed"),
    )

    # Scheduler unavailable tests
    test_schedule_scheduler_unavailable = factory.make_schedule_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=500,
        expected_error_substring="not available",  # Matches "Prometheus scheduler not available"
        mock_scheduler_none=True,
    )

    test_delete_scheduler_unavailable = factory.make_delete_endpoint_error_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,
        expected_error_substring="not available",  # Matches "Prometheus scheduler not available"
        mock_scheduler_none=True,
    )

    # ========================================================================
    # Additional Coverage Tests
    # ========================================================================

    # Test with custom threshold_delta and epsilon (covers custom alpha and epsilon)
    test_compute_custom_threshold_and_epsilon = factory.make_compute_endpoint_test(
        metric_name="KSTestStreaming",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/ksteststreaming",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "thresholdDelta": 0.01,  # Custom alpha value
            "epsilon": 0.005,  # Custom epsilon value
            "batchSize": 50,  # Also test custom batch size
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha", "epsilon"],
        df_type="Polars",
    )

    # ========================================================================
    # ApproxKSTestMetricRequest.retrieve_tags() Tests
    # ========================================================================

    def test_retrieve_tags_with_all_fields(self):
        """Test retrieve_tags method with all fields populated."""
        request = ApproxKSTestMetricRequest(
            modelId="test-model",
            referenceTag="baseline",
            fitColumns=["feature1", "feature2"],
            epsilon=0.01,
        )

        tags = request.retrieve_tags()

        # Check that tags include the base tags plus KSTestStreaming-specific tags
        assert "modelId" in tags
        assert tags["modelId"] == "test-model"
        assert "referenceTag" in tags
        assert tags["referenceTag"] == "baseline"
        assert "fitColumns" in tags
        assert tags["fitColumns"] == "feature1,feature2"
        assert "epsilon" in tags
        assert tags["epsilon"] == "0.01"

    def test_retrieve_tags_without_reference_tag(self):
        """Test retrieve_tags method without referenceTag."""
        request = ApproxKSTestMetricRequest(
            modelId="test-model",
            fitColumns=["feature1"],
            epsilon=0.01,
        )

        tags = request.retrieve_tags()

        # Check that tags include base tags but not referenceTag
        assert "modelId" in tags
        assert tags["modelId"] == "test-model"
        assert "referenceTag" not in tags
        assert "fitColumns" in tags
        assert "epsilon" in tags

    def test_retrieve_tags_without_fit_columns(self):
        """Test retrieve_tags method without fitColumns."""
        request = ApproxKSTestMetricRequest(
            modelId="test-model",
            referenceTag="baseline",
            epsilon=0.01,
        )

        tags = request.retrieve_tags()

        # Check that tags include referenceTag and epsilon but not fitColumns
        assert "modelId" in tags
        assert "referenceTag" in tags
        assert "fitColumns" not in tags
        assert "epsilon" in tags

    def test_retrieve_tags_with_default_epsilon(self):
        """Test retrieve_tags method with default epsilon value."""
        request = ApproxKSTestMetricRequest(
            modelId="test-model",
            referenceTag="baseline",
            fitColumns=["feature1"],
        )

        tags = request.retrieve_tags()

        # Check that tags include epsilon with default value
        assert "epsilon" in tags
        assert tags["epsilon"] == "0.01"  # Default epsilon


# ============================================================================
# DEPRECATED ENDPOINTS TESTS (ApproxKSTest -> KSTestStreaming)
# ============================================================================


class TestApproxKSTestDeprecatedEndpoints:
    """Tests for deprecated ApproxKSTest endpoints (backwards compatibility)."""

    # Test deprecated compute endpoint forwards to new endpoint
    test_deprecated_compute_endpoint = factory.make_compute_endpoint_test(
        metric_name="KSTestStreaming",  # Still uses new metric name internally
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/approxkstest",  # Deprecated path
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "epsilon": 0.01,
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha", "epsilon"],
        df_type="Pandas",
    )

    # Test deprecated definition endpoint forwards to new endpoint
    test_deprecated_definition_endpoint = factory.make_definition_endpoint_test(
        metric_name="KSTestStreaming",
        endpoint_path="/metrics/drift/approxkstest/definition",  # Deprecated path
        client=client,
        expected_name="Kolmogorov-Smirnov",
    )

    # Test deprecated schedule endpoint forwards to new endpoint
    test_deprecated_schedule_endpoint = factory.make_schedule_endpoint_test(
        metric_name="KSTestStreaming",  # Still uses new metric name internally
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/approxkstest/request",  # Deprecated path
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "epsilon": 0.01,
        },
    )

    # Test deprecated delete endpoint forwards to new endpoint
    test_deprecated_delete_endpoint = factory.make_delete_schedule_endpoint_test(
        metric_name="KSTestStreaming",  # Still uses new metric name internally
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/approxkstest/request",  # Deprecated path
        client=client,
    )

    # Test deprecated list requests endpoint forwards to new endpoint
    test_deprecated_list_requests_endpoint = factory.make_list_requests_endpoint_test(
        metric_name="KSTestStreaming",  # Still uses new metric name internally
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov_streaming",
        endpoint_path="/metrics/drift/approxkstest/requests",  # Deprecated path
        client=client,
    )


# ============================================================================
# Multi-Feature Aggregation Tests
# ============================================================================


def test_ksteststreaming_multi_feature_aggregation(monkeypatch):
    """Ensure multi-feature responses include per-feature results and correct aggregation."""
    from unittest.mock import AsyncMock, MagicMock

    import pandas as pd

    # Mock data source to return test dataframes
    mock_data_source = MagicMock()

    # Create test dataframes with multiple features
    ref_df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [10.0, 20.0, 30.0, 40.0]})

    cur_df = pd.DataFrame({"feature1": [1.5, 2.5, 3.5, 4.5], "feature2": [15.0, 25.0, 35.0, 45.0]})

    mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=ref_df)
    mock_data_source.get_organic_dataframe = AsyncMock(return_value=cur_df)

    # Patch the get_data_source function
    def mock_get_data_source():
        return mock_data_source

    import src.endpoints.metrics.drift.kolmogorov_smirnov_streaming as ks_module

    monkeypatch.setattr(ks_module, "get_data_source", mock_get_data_source)

    payload = {
        "modelId": "test-model",
        "referenceTag": "baseline",
        "fitColumns": ["feature1", "feature2"],
        "batchSize": 100,
        "epsilon": 0.01,
    }

    response = client.post("/metrics/drift/ksteststreaming", json=payload)
    assert response.status_code == 200

    body = response.json()

    # feature_results exists and has one entry per feature
    assert "feature_results" in body
    feature_results = body["feature_results"]
    assert isinstance(feature_results, dict)
    assert len(feature_results) == len(payload["fitColumns"])

    # each entry has expected keys
    expected_feature_keys = {"statistic", "p_value", "drift_detected", "n_reference", "n_current", "epsilon", "alpha"}
    for feature_name in payload["fitColumns"]:
        assert feature_name in feature_results
        result = feature_results[feature_name]
        assert expected_feature_keys.issubset(result.keys())

    # top-level value and p_value aggregate per-feature statistics
    max_statistic = max(r["statistic"] for r in feature_results.values())
    min_p_value = min(r["p_value"] for r in feature_results.values())

    assert body["value"] == max_statistic
    assert body["p_value"] == min_p_value
