from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.drift.kolmogorov_smirnov import KSTestMetricRequest, router

from . import factory

# Create test app with KS Test router
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestKSTestEndpoints:
    """Unified endpoint tests for KS Test metric."""

    # Pandas DataFrame tests
    test_compute_endpoint_pandas = factory.make_compute_endpoint_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha"],
        df_type="Pandas",
    )

    # Polars DataFrame tests
    test_compute_endpoint_polars = factory.make_compute_endpoint_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha"],
        df_type="Polars",
    )

    test_definition_endpoint = factory.make_definition_endpoint_test(
        metric_name="KSTest",
        endpoint_path="/metrics/drift/kstest/definition",
        client=client,
        expected_name="Kolmogorov-Smirnov",
    )

    test_schedule_endpoint = factory.make_schedule_endpoint_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    test_delete_schedule_endpoint = factory.make_delete_schedule_endpoint_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
        client=client,
    )

    test_list_requests_endpoint = factory.make_list_requests_endpoint_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/requests",
        client=client,
    )

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    # Compute endpoint errors
    test_compute_missing_reference_tag = factory.make_compute_endpoint_error_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest",
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
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest",
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
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest",
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
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
        client=client,
        request_id="not-a-valid-uuid",
        expected_status_code=500,  # Endpoint catches ValueError and returns 500
        expected_error_substring="Invalid request ID",
    )

    # List endpoint with multiple requests
    test_list_requests_with_data = factory.make_list_requests_with_data_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/requests",
        client=client,
        num_requests=3,
    )

    # List endpoint with malformed requests (defensive logic test)
    test_list_requests_filters_malformed = factory.make_list_requests_with_malformed_data_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/requests",
        client=client,
        num_valid_requests=2,
        num_malformed_requests=3,
    )

    # ========================================================================
    # Scheduler Exception Tests
    # ========================================================================

    # Schedule endpoint scheduler exceptions
    test_schedule_connection_error = factory.make_schedule_endpoint_error_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
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
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
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
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
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
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,
        expected_error_substring="connect",  # Match "Failed to connect"
        mock_scheduler_none=False,
        delete_side_effect=ConnectionError("Failed to connect to scheduler database"),
    )

    test_delete_runtime_error = factory.make_delete_endpoint_error_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,
        expected_error_substring="failed",
        mock_scheduler_none=False,
        delete_side_effect=RuntimeError("Scheduler deletion failed"),
    )

    # Scheduler unavailable tests
    # Note: Current endpoint implementation returns 500 instead of 503 for scheduler unavailable
    # This could be improved to return 503 (Service Unavailable) in a future update
    test_schedule_scheduler_unavailable = factory.make_schedule_endpoint_error_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=500,  # TODO: Should be 503 Service Unavailable
        expected_error_substring="not available",  # Matches "Prometheus scheduler not available"
        mock_scheduler_none=True,
    )

    test_delete_scheduler_unavailable = factory.make_delete_endpoint_error_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,  # TODO: Should be 503 Service Unavailable
        expected_error_substring="not available",  # Matches "Prometheus scheduler not available"
        mock_scheduler_none=True,
    )

    # ========================================================================
    # Empty Data Tests
    # ========================================================================

    test_compute_empty_reference_data = factory.make_compute_empty_reference_data_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    test_compute_empty_current_data = factory.make_compute_empty_current_data_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    # ========================================================================
    # List Endpoint Exception Tests
    # ========================================================================

    test_list_scheduler_unavailable = factory.make_list_endpoint_scheduler_unavailable_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/requests",
        client=client,
    )

    test_list_exception_handling = factory.make_list_endpoint_exception_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest/requests",
        client=client,
    )

    # ========================================================================
    # Compute Endpoint Generic Exception
    # ========================================================================

    test_compute_generic_exception = factory.make_compute_generic_exception_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    # ========================================================================
    # Additional Coverage Tests
    # ========================================================================

    # Test with custom threshold_delta (covers line 84: custom alpha)
    test_compute_custom_threshold = factory.make_compute_endpoint_test(
        metric_name="KSTest",
        module_path="src.endpoints.metrics.drift.kolmogorov_smirnov",
        endpoint_path="/metrics/drift/kstest",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "thresholdDelta": 0.01,  # Custom alpha value
            "batchSize": 50,  # Also test custom batch size (line 63)
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha"],
        df_type="Polars",
    )

    # ========================================================================
    # KSTestMetricRequest.retrieve_tags() Tests
    # ========================================================================

    def test_retrieve_tags_with_all_fields(self):
        """Test retrieve_tags method with all fields populated."""
        request = KSTestMetricRequest(
            modelId="test-model",
            referenceTag="baseline",
            fitColumns=["feature1", "feature2"],
        )

        tags = request.retrieve_tags()

        # Check that tags include the base tags plus KSTest-specific tags
        assert "modelId" in tags
        assert tags["modelId"] == "test-model"
        assert "referenceTag" in tags
        assert tags["referenceTag"] == "baseline"
        assert "fitColumns" in tags
        assert tags["fitColumns"] == "feature1,feature2"

    def test_retrieve_tags_without_reference_tag(self):
        """Test retrieve_tags method without referenceTag."""
        request = KSTestMetricRequest(
            modelId="test-model",
            fitColumns=["feature1"],
        )

        tags = request.retrieve_tags()

        # Check that tags include base tags but not referenceTag
        assert "modelId" in tags
        assert tags["modelId"] == "test-model"
        assert "referenceTag" not in tags
        assert "fitColumns" in tags

    def test_retrieve_tags_without_fit_columns(self):
        """Test retrieve_tags method without fitColumns."""
        request = KSTestMetricRequest(
            modelId="test-model",
            referenceTag="baseline",
        )

        tags = request.retrieve_tags()

        # Check that tags include referenceTag but not fitColumns
        assert "modelId" in tags
        assert "referenceTag" in tags
        assert "fitColumns" not in tags
