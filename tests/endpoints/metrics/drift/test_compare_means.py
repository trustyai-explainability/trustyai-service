from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd

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
    # Compute Endpoint Logic Tests
    # ========================================================================

    def test_multi_feature_drift_consistency(self):
        """
        Test that when drift is detected, the returned p_value is consistent with drift_detected.
        
        This test verifies the fix for the issue where drift_detected=True but p_value > alpha
        when multiple features are tested and the feature with max statistic didn't detect drift.
        """
        # Create a scenario where:
        # - Feature A: large statistic but p_value > alpha (no drift)
        # - Feature B: smaller statistic but p_value < alpha (drift detected)
        # The old code would return Feature A's p_value, causing inconsistency
        
        # Mock data that will produce the desired scenario
        # We'll use actual t-test results by creating data with known properties
        np.random.seed(42)
        
        # Feature A: large difference but high variance -> large statistic, high p-value
        ref_a = np.random.normal(0, 1, 100)
        cur_a = np.random.normal(0.1, 3, 100)  # Small mean shift, large variance
        
        # Feature B: smaller difference but low variance -> smaller statistic, low p-value
        ref_b = np.random.normal(0, 1, 100)
        cur_b = np.random.normal(0.5, 1, 100)  # Larger mean shift, same variance
        
        reference_df = pd.DataFrame({
            "featureA": ref_a,
            "featureB": ref_b,
        })
        current_df = pd.DataFrame({
            "featureA": cur_a,
            "featureB": cur_b,
        })
        
        with patch("src.endpoints.metrics.drift.compare_means.get_data_source") as mock_ds:
            mock_data_source = MagicMock()
            mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=reference_df)
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=current_df)
            mock_ds.return_value = mock_data_source
            
            # Make request with both features
            response = client.post(
                "/metrics/drift/comparemeans",
                json={
                    "modelId": "test-model",
                    "referenceTag": "baseline",
                    "fitColumns": ["featureA", "featureB"],
                    "alpha": 0.05,
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify consistency: if drift_detected=True, then p_value < alpha
            if data["drift_detected"]:
                assert data["p_value"] < data["alpha"], (
                    f"Inconsistency: drift_detected=True but p_value={data['p_value']} >= alpha={data['alpha']}"
                )
            else:
                # If no drift detected, p_value should be >= alpha (or at least not contradict)
                # Note: This is less strict since we might return the max statistic feature
                pass
            
            # Verify feature_results contains both features
            assert "feature_results" in data
            assert "featureA" in data["feature_results"]
            assert "featureB" in data["feature_results"]
            
            # Verify that at least one feature detected drift (since we set up the scenario)
            feature_drift_detected = [
                data["feature_results"][f]["drift_detected"]
                for f in ["featureA", "featureB"]
            ]
            assert any(feature_drift_detected), "At least one feature should detect drift in this test scenario"

    def test_multi_feature_drift_consistency_deterministic(self):
        """
        Deterministic test for drift consistency fix.
        
        Tests the specific bug: when Feature A has max statistic but no drift,
        and Feature B has drift, the returned p_value should come from Feature B
        (a drifting feature), not Feature A.
        """
        # Create a scenario where we know the outcomes:
        # Feature A: small mean shift, high variance -> large statistic but high p-value (no drift)
        # Feature B: large mean shift, low variance -> smaller statistic but low p-value (drift)
        np.random.seed(123)
        ref_a = np.random.normal(0, 1, 50)
        cur_a = np.random.normal(0.1, 3, 50)  # Small shift, high variance -> high p-value
        
        ref_b = np.random.normal(0, 1, 50)
        cur_b = np.random.normal(1.5, 1, 50)  # Large shift, same variance -> low p-value
        
        reference_df = pd.DataFrame({"featureA": ref_a, "featureB": ref_b})
        current_df = pd.DataFrame({"featureA": cur_a, "featureB": cur_b})
        
        with patch("src.endpoints.metrics.drift.compare_means.get_data_source") as mock_ds:
            mock_data_source = MagicMock()
            mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=reference_df)
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=current_df)
            mock_ds.return_value = mock_data_source
            
            response = client.post(
                "/metrics/drift/comparemeans",
                json={
                    "modelId": "test-model",
                    "referenceTag": "baseline",
                    "fitColumns": ["featureA", "featureB"],
                    "alpha": 0.05,
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # The key assertion: if drift_detected=True, p_value MUST be < alpha
            if data["drift_detected"]:
                assert data["p_value"] < data["alpha"], (
                    f"BUG: drift_detected=True but p_value={data['p_value']} >= alpha={data['alpha']}. "
                    f"This indicates the p_value came from a non-drifting feature."
                )
            
            # Additional check: verify the p_value comes from a drifting feature
            if data["drift_detected"]:
                drifting_features = [
                    f for f, r in data["feature_results"].items()
                    if r["drift_detected"]
                ]
                # The returned p_value should match one of the drifting features' p_values
                drifting_p_values = [
                    data["feature_results"][f]["p_value"]
                    for f in drifting_features
                ]
                assert data["p_value"] in drifting_p_values, (
                    f"Returned p_value={data['p_value']} doesn't match any drifting feature's p_value: {drifting_p_values}"
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

    def test_retrieve_default_tags_with_none_metric_name(self):
        """Test that metric_name is automatically set via model_validator (fixes type violation issue)."""
        # Create request without metric_name (it defaults to None)
        request = CompareMeansMetricRequest(
            modelId="test-model",
        )

        # Model validator should have set metric_name automatically during initialization
        assert request.metric_name == "CompareMeans"
        
        # This should not raise an error and should not add None to Dict[str, str]
        tags = request.retrieve_default_tags()
        
        # Verify tags are all strings (no None values)
        assert "modelId" in tags
        assert tags["modelId"] == "test-model"
        assert "metricName" in tags
        assert tags["metricName"] == "CompareMeans"
        assert isinstance(tags["metricName"], str)
        assert isinstance(tags["modelId"], str)
        
        # Verify all values are strings (not None)
        for key, value in tags.items():
            assert value is not None, f"Tag {key} should not be None"
            assert isinstance(value, str), f"Tag {key} should be str, got {type(value)}"

    def test_retrieve_default_tags_called_directly_by_prometheus_publisher(self):
        """
        Test that retrieve_default_tags() works when called directly (as prometheus_publisher does).
        
        This simulates the actual issue: prometheus_publisher._generate_tags() calls
        retrieve_default_tags() directly, bypassing retrieve_tags(). The model_validator
        ensures metric_name is set during initialization, so this won't add None to Dict[str, str].
        """
        from prometheus_client import CollectorRegistry
        from src.service.prometheus.prometheus_publisher import PrometheusPublisher
        import uuid

        # Create request without metric_name (it defaults to None, but model_validator sets it)
        request = CompareMeansMetricRequest(
            modelId="test-model",
        )

        # Model validator should have set metric_name automatically
        assert request.metric_name == "CompareMeans"
        
        # Simulate what prometheus_publisher does: call retrieve_default_tags() directly
        # This should not raise an error and should not add None to the dict
        tags_from_default = request.retrieve_default_tags()
        
        # Verify all values are strings (not None)
        for key, value in tags_from_default.items():
            assert value is not None, f"Tag {key} should not be None (would violate Dict[str, str])"
            assert isinstance(value, str), f"Tag {key} should be str, got {type(value)}"
        
        # Now test with actual prometheus_publisher to ensure it works end-to-end
        registry = CollectorRegistry()
        publisher = PrometheusPublisher(registry=registry)
        test_id = uuid.uuid4()
        
        # This should not raise an error
        publisher.gauge(
            model_name="test_model",
            id=test_id,
            value=0.5,
            request=request
        )
        
        # Verify the gauge was created successfully
        metric_name = f"trustyai_{request.metric_name.lower()}"
        assert metric_name in publisher.registry._names_to_collectors
