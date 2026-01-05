from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.drift.kolmogorov_smirnov import router

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
        df_type="pandas",
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
        df_type="polars",
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
