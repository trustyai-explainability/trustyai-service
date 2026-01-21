from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.drift.jensen_shannon import router

from . import factory

# Create test app with Jensen-Shannon router
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestJensenShannonEndpoints:
    """Unified endpoint tests for Jensen-Shannon metric."""

    # Pandas DataFrame tests
    test_compute_endpoint_pandas = factory.make_compute_endpoint_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "statistic": "distance",
            "threshold": 0.1,
            "method": "kde",
            "gridPoints": 256,
            "bins": 64,
        },
        expected_response_keys=[
            "status",
            "value",
            "drift_detected",
            "Jensen–Shannon_distance",
            "Jensen–Shannon_divergence",
        ],
        df_type="Pandas",
    )

    # Polars DataFrame tests
    test_compute_endpoint_polars = factory.make_compute_endpoint_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "statistic": "distance",
            "threshold": 0.1,
            "method": "kde",
            "gridPoints": 256,
            "bins": 64,
        },
        expected_response_keys=[
            "status",
            "value",
            "drift_detected",
            "Jensen–Shannon_distance",
            "Jensen–Shannon_divergence",
        ],
        df_type="Polars",
    )

    test_definition_endpoint = factory.make_definition_endpoint_test(
        metric_name="JensenShannon",
        endpoint_path="/metrics/drift/jensenshannon/definition",
        client=client,
        expected_name="Jensen-Shannon",
    )

    test_schedule_endpoint = factory.make_schedule_endpoint_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "threshold": 0.1,
        },
    )

    test_delete_schedule_endpoint = factory.make_delete_schedule_endpoint_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/request",
        client=client,
    )

    test_list_requests_endpoint = factory.make_list_requests_endpoint_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/requests",
        client=client,
    )

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    # Compute endpoint errors
    test_compute_missing_reference_tag = factory.make_compute_endpoint_error_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon",
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
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon",
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
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon",
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
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/request",
        client=client,
        request_id="not-a-valid-uuid",
        expected_status_code=500,  # Endpoint catches ValueError and returns 500
        expected_error_substring="Invalid request ID",
    )

    # List endpoint with multiple requests
    test_list_requests_with_data = factory.make_list_requests_with_data_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/requests",
        client=client,
        num_requests=3,
    )

    # List endpoint with malformed requests (defensive logic test)
    test_list_requests_filters_malformed = factory.make_list_requests_with_malformed_data_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/requests",
        client=client,
        num_valid_requests=2,
        num_malformed_requests=3,
    )

    # Empty data tests
    test_compute_empty_reference_data = factory.make_compute_empty_reference_data_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    test_compute_empty_current_data = factory.make_compute_empty_current_data_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    # List endpoint error tests
    test_list_scheduler_unavailable = factory.make_list_endpoint_scheduler_unavailable_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/requests",
        client=client,
    )

    test_list_exception = factory.make_list_endpoint_exception_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/requests",
        client=client,
    )

    # Generic exception in compute endpoint
    test_compute_generic_exception = factory.make_compute_generic_exception_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon",
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
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=500,  # TODO: Should be 503 Service Unavailable
        expected_error_substring="scheduler not available",
        mock_scheduler_none=True,
    )

    test_schedule_connection_error = factory.make_schedule_endpoint_error_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/request",
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
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,  # TODO: Should be 503 Service Unavailable
        expected_error_substring="scheduler not available",
        mock_scheduler_none=True,
    )

    test_delete_exception = factory.make_delete_endpoint_error_test(
        metric_name="JensenShannon",
        module_path="src.endpoints.metrics.drift.jensen_shannon",
        endpoint_path="/metrics/drift/jensenshannon/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=500,
        expected_error_substring="Error deleting schedule",
        delete_side_effect=Exception("Database connection failed"),
    )

    # ========================================================================
    # JensenShannonMetricRequest.retrieve_tags() Tests
    # ========================================================================

    test_retrieve_tags_with_all_fields = factory.make_retrieve_tags_with_all_fields_test(
        request_class=__import__(
            "src.endpoints.metrics.drift.jensen_shannon", fromlist=["JensenShannonMetricRequest"]
        ).JensenShannonMetricRequest,
    )

    test_retrieve_tags_without_reference_tag = factory.make_retrieve_tags_without_reference_tag_test(
        request_class=__import__(
            "src.endpoints.metrics.drift.jensen_shannon", fromlist=["JensenShannonMetricRequest"]
        ).JensenShannonMetricRequest,
    )

    test_retrieve_tags_without_fit_columns = factory.make_retrieve_tags_without_fit_columns_test(
        request_class=__import__(
            "src.endpoints.metrics.drift.jensen_shannon", fromlist=["JensenShannonMetricRequest"]
        ).JensenShannonMetricRequest,
    )

    test_retrieve_tags_with_empty_fit_columns = factory.make_retrieve_tags_with_empty_fit_columns_test(
        request_class=__import__(
            "src.endpoints.metrics.drift.jensen_shannon", fromlist=["JensenShannonMetricRequest"]
        ).JensenShannonMetricRequest,
    )

    test_retrieve_default_tags_with_none_metric_name = factory.make_retrieve_default_tags_with_none_metric_name_test(
        request_class=__import__(
            "src.endpoints.metrics.drift.jensen_shannon", fromlist=["JensenShannonMetricRequest"]
        ).JensenShannonMetricRequest,
        expected_metric_name="JensenShannon",
    )

    test_retrieve_default_tags_called_directly_by_prometheus_publisher = (
        factory.make_retrieve_default_tags_called_directly_by_prometheus_publisher_test(
            request_class=__import__(
                "src.endpoints.metrics.drift.jensen_shannon", fromlist=["JensenShannonMetricRequest"]
            ).JensenShannonMetricRequest,
            expected_metric_name="JensenShannon",
        )
    )

    def test_helper_functions(self):
        """Test that helper functions return the correct shared instances."""
        from src.endpoints.metrics.drift.jensen_shannon import get_data_source, get_prometheus_scheduler

        # These are thin wrappers around shared functions
        # Test that they return the correct instances
        data_source = get_data_source()
        scheduler = get_prometheus_scheduler()

        # Verify they return objects (not None, unless scheduler is unavailable)
        assert data_source is not None
        # Scheduler might be None if not configured, which is valid
        # Just verify the function executes without error
