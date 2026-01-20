from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.drift.compare_means import CompareMeansMetricRequest, router

from . import factory

# Create test app with CompareMeans router
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestCompareMeansEndpoints:
    """Unified endpoint tests for CompareMeans metric."""

    # Pandas DataFrame tests
    test_compute_endpoint_pandas = factory.make_compute_endpoint_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "alpha": 0.05,
            "equalVar": False,
            "nanPolicy": "omit",
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha"],
        df_type="Pandas",
    )

    # Polars DataFrame tests
    test_compute_endpoint_polars = factory.make_compute_endpoint_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "alpha": 0.05,
            "equalVar": False,
            "nanPolicy": "omit",
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha"],
        df_type="Polars",
    )

    test_definition_endpoint = factory.make_definition_endpoint_test(
        metric_name="CompareMeans",
        endpoint_path="/metrics/drift/comparemeans/definition",
        client=client,
        expected_name="T-Test",
    )

    test_schedule_endpoint = factory.make_schedule_endpoint_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "alpha": 0.05,
        },
    )

    test_delete_schedule_endpoint = factory.make_delete_schedule_endpoint_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/request",
        client=client,
    )

    test_list_requests_endpoint = factory.make_list_requests_endpoint_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/requests",
        client=client,
    )

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    # Compute endpoint errors
    test_compute_missing_reference_tag = factory.make_compute_endpoint_error_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans",
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
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans",
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
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans",
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
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/request",
        client=client,
        request_id="not-a-valid-uuid",
        expected_status_code=500,  # Endpoint catches ValueError and returns 500
        expected_error_substring="Invalid request ID",
    )

    # List endpoint with multiple requests
    test_list_requests_with_data = factory.make_list_requests_with_data_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/requests",
        client=client,
        num_requests=3,
    )

    # List endpoint with malformed requests (defensive logic test)
    test_list_requests_filters_malformed = factory.make_list_requests_with_malformed_data_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/requests",
        client=client,
        num_valid_requests=2,
        num_malformed_requests=2,
    )

    # Empty data tests
    test_compute_empty_reference_data = factory.make_compute_empty_reference_data_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    test_compute_empty_current_data = factory.make_compute_empty_current_data_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    # List endpoint error tests
    test_list_scheduler_unavailable = factory.make_list_endpoint_scheduler_unavailable_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/requests",
        client=client,
    )

    test_list_exception = factory.make_list_endpoint_exception_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/requests",
        client=client,
    )

    # Generic exception in compute endpoint
    test_compute_generic_exception = factory.make_compute_generic_exception_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    # ========================================================================
    # Scheduler Error Tests
    # ========================================================================

    test_schedule_scheduler_unavailable = factory.make_schedule_endpoint_error_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=500,
        expected_error_substring="scheduler not available",
        mock_scheduler_none=True,
    )

    test_schedule_register_exception = factory.make_schedule_endpoint_error_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=500,
        expected_error_substring="Error scheduling metric",
        register_side_effect=Exception("Database connection failed"),
    )

    test_delete_scheduler_unavailable = factory.make_delete_endpoint_error_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,
        expected_error_substring="scheduler not available",
        mock_scheduler_none=True,
    )

    test_delete_exception = factory.make_delete_endpoint_error_test(
        metric_name="CompareMeans",
        module_path="src.endpoints.metrics.drift.compare_means",
        endpoint_path="/metrics/drift/comparemeans/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,
        expected_error_substring="Error deleting schedule",
        delete_side_effect=Exception("Database connection failed"),
    )

    # ========================================================================
    # Deprecated Meanshift Endpoint Tests
    # ========================================================================

    test_deprecated_compute_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="Meanshift",
        deprecated_endpoint_path="/metrics/drift/meanshift",
        client=client,
        endpoint_type="compute",
        module_path="src.endpoints.metrics.drift.compare_means",
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "alpha": 0.05,
            "equalVar": False,
            "nanPolicy": "omit",
        },
        expected_response_keys=["status", "value", "drift_detected", "p_value", "alpha"],
    )

    test_deprecated_definition_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="Meanshift",
        deprecated_endpoint_path="/metrics/drift/meanshift/definition",
        client=client,
        endpoint_type="definition",
        expected_name_substring="T-Test",
    )

    test_deprecated_schedule_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="Meanshift",
        deprecated_endpoint_path="/metrics/drift/meanshift/request",
        client=client,
        endpoint_type="schedule",
        module_path="src.endpoints.metrics.drift.compare_means",
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "alpha": 0.05,
        },
    )

    test_deprecated_delete_schedule_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="Meanshift",
        deprecated_endpoint_path="/metrics/drift/meanshift/request",
        client=client,
        endpoint_type="delete",
        module_path="src.endpoints.metrics.drift.compare_means",
    )

    test_deprecated_list_requests_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="Meanshift",
        deprecated_endpoint_path="/metrics/drift/meanshift/requests",
        client=client,
        endpoint_type="list",
        module_path="src.endpoints.metrics.drift.compare_means",
    )

    # ========================================================================
    # CompareMeansMetricRequest.retrieve_tags() Tests
    # ========================================================================

    def test_retrieve_tags_with_all_fields(self):
        """Test retrieve_tags method with all fields populated."""
        request = CompareMeansMetricRequest(
            modelId="test-model",
            referenceTag="baseline",
            fitColumns=["feature1", "feature2"],
        )

        tags = request.retrieve_tags()

        # Check that tags include the base tags plus CompareMeans-specific tags
        assert "modelId" in tags
        assert tags["modelId"] == "test-model"
        assert "referenceTag" in tags
        assert tags["referenceTag"] == "baseline"
        assert "fitColumns" in tags
        assert tags["fitColumns"] == "feature1,feature2"

    def test_retrieve_tags_without_reference_tag(self):
        """Test retrieve_tags method without referenceTag."""
        request = CompareMeansMetricRequest(
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
        request = CompareMeansMetricRequest(
            modelId="test-model",
            referenceTag="baseline",
        )

        tags = request.retrieve_tags()

        # Check that tags include referenceTag but not fitColumns
        assert "modelId" in tags
        assert "referenceTag" in tags
        assert "fitColumns" not in tags
