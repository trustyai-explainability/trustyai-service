"""
Unified tests for all drift metric endpoints.

This module provides factory functions to create common tests that apply to all
drift metric endpoints, similar to the pattern in tests/metrics/test_drift_unified.py.

All drift metric endpoints share a common structure:
- POST /metrics/drift/{metric} - Compute metric
- GET /metrics/drift/{metric}/definition - Get metric definition
- POST /metrics/drift/{metric}/request - Schedule recurring computation
- DELETE /metrics/drift/{metric}/request - Delete scheduled computation
- GET /metrics/drift/{metric}/requests - List scheduled computations

Common request fields:
- modelId: Model identifier
- requestName: Optional request name
- metricName: Optional metric name (often set automatically)
- batchSize: Optional batch size for current data
- thresholdDelta: Optional threshold for drift detection
- referenceTag: Optional tag for reference data
- fitColumns: List of feature columns to analyze
"""

from typing import Any, Callable, Dict, List, Literal, Union
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl

# ============================================================================
# Factory Functions for Common Tests
# ============================================================================


def make_compute_endpoint_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    request_payload: Dict[str, Any],
    expected_response_keys: List[str],
    df_type: Literal["Pandas", "Polars"] = "Polars",
) -> Callable[[], None]:
    """
    Factory to create a test for the compute endpoint.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching (e.g., "src.endpoints.metrics.drift.kolmogorov_smirnov")
    :param endpoint_path: API endpoint path (e.g., "/metrics/drift/kstest")
    :param client: TestClient instance for making requests
    :param request_payload: Request payload dictionary
    :param expected_response_keys: Keys expected in successful response
    :param df_type: Type of DataFrame to use ("Pandas" or "Polars")
    :return: Test function
    """

    @patch(f"{module_path}.get_data_source")
    def test_impl(self: Any, mock_ds: MagicMock) -> None:
        """Test compute endpoint returns valid response structure."""
        # Create sample dataframe (Pandas or Polars based on df_type)
        sample_df = _create_sample_dataframe(request_payload.get("fitColumns", ["feature1"]), df_type=df_type)

        # Mock data source
        mock_data_source = MagicMock()
        mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=sample_df)
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
        mock_ds.return_value = mock_data_source

        # Send request
        response = client.post(endpoint_path, json=request_payload)

        # Verify response
        assert response.status_code == 200, f"{metric_name} compute failed: {response.text}"
        data = response.json()

        # Check all expected keys are present
        for key in expected_response_keys:
            assert key in data, f"Missing key '{key}' in {metric_name} response"

        # Common validations
        if "status" in data:
            assert data["status"] == "success"
        if "drift_detected" in data:
            assert isinstance(data["drift_detected"], bool)
        if "value" in data:
            assert isinstance(data["value"], (int, float))

    return test_impl


def make_definition_endpoint_test(
    metric_name: str,
    endpoint_path: str,
    client: Any,
    expected_name: str,
) -> Callable[[], None]:
    """
    Factory to create a test for the definition endpoint.

    :param metric_name: Name of the metric for logging
    :param endpoint_path: API endpoint path (e.g., "/metrics/drift/kstest/definition")
    :param client: TestClient instance for making requests
    :param expected_name: Expected metric name in response
    :return: Test function
    """

    def test_impl(self: Any) -> None:
        """Test definition endpoint returns metric name and description."""
        response = client.get(endpoint_path)

        assert response.status_code == 200, f"{metric_name} definition failed: {response.text}"
        data = response.json()

        # Verify structure
        assert "name" in data, f"Missing 'name' in {metric_name} definition"
        assert "description" in data, f"Missing 'description' in {metric_name} definition"

        # Verify name matches expected
        assert expected_name.lower() in data["name"].lower(), (
            f"Expected name '{expected_name}' not found in {metric_name} definition name: {data['name']}"
        )

        # Verify description is non-empty
        assert len(data["description"]) > 0, f"{metric_name} definition description is empty"

    return test_impl


def make_schedule_endpoint_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    request_payload: Dict[str, Any],
) -> Callable[[], None]:
    """
    Factory to create a test for the schedule/request endpoint.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching (e.g., "src.endpoints.metrics.drift.kolmogorov_smirnov")
    :param endpoint_path: API endpoint path (e.g., "/metrics/drift/kstest/request")
    :param client: TestClient instance for making requests
    :param request_payload: Request payload dictionary
    :return: Test function
    """

    @patch(f"{module_path}.get_prometheus_scheduler")
    @patch(f"{module_path}.get_data_source")
    def test_impl(self: Any, mock_ds: MagicMock, mock_sched_fn: MagicMock) -> None:
        """Test schedule endpoint returns requestId."""
        # Mock scheduler
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        # Mock data source (needed for KS test registration)
        mock_data_source = MagicMock()
        mock_data_source.get_metadata = AsyncMock(return_value={"feature1": "type1"})
        mock_ds.return_value = mock_data_source

        # Send request
        response = client.post(endpoint_path, json=request_payload)

        # Verify response
        assert response.status_code == 200, f"{metric_name} schedule failed: {response.text}"
        data = response.json()

        # Verify requestId is present and valid UUID format
        assert "requestId" in data, f"Missing 'requestId' in {metric_name} schedule response"
        request_id = data["requestId"]
        assert isinstance(request_id, str), f"{metric_name} requestId is not a string"
        assert len(request_id) > 0, f"{metric_name} requestId is empty"

        # Verify it looks like a UUID (basic check)
        assert "-" in request_id, f"{metric_name} requestId doesn't look like a UUID: {request_id}"

    return test_impl


def make_delete_schedule_endpoint_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
) -> Callable[[], None]:
    """
    Factory to create a test for the delete schedule endpoint.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching (e.g., "src.endpoints.metrics.drift.kolmogorov_smirnov")
    :param endpoint_path: API endpoint path (e.g., "/metrics/drift/kstest/request")
    :param client: TestClient instance for making requests
    :return: Test function
    """

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(self: Any, mock_sched_fn: MagicMock) -> None:
        """Test delete schedule endpoint."""
        # Mock scheduler
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        # Test UUID
        test_uuid = "123e4567-e89b-12d3-a456-426614174000"
        payload = {"requestId": test_uuid}

        # Send request (DELETE uses request() with json parameter)
        response = client.request("DELETE", endpoint_path, json=payload)

        # Verify response
        assert response.status_code == 200, f"{metric_name} delete schedule failed: {response.text}"
        data = response.json()

        # Verify success status
        assert "status" in data, f"Missing 'status' in {metric_name} delete response"
        assert data["status"] == "success", f"{metric_name} delete did not return success status"

    return test_impl


def make_list_requests_endpoint_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
) -> Callable[[], None]:
    """
    Factory to create a test for the list requests endpoint.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching (e.g., "src.endpoints.metrics.drift.kolmogorov_smirnov")
    :param endpoint_path: API endpoint path (e.g., "/metrics/drift/kstest/requests")
    :param client: TestClient instance for making requests
    :return: Test function
    """

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(self: Any, mock_sched_fn: MagicMock) -> None:
        """Test list requests endpoint returns requests array."""
        # Mock scheduler
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        # Send request
        response = client.get(endpoint_path)

        # Verify response
        assert response.status_code == 200, f"{metric_name} list requests failed: {response.text}"
        data = response.json()

        # Verify requests array is present
        assert "requests" in data, f"Missing 'requests' in {metric_name} list response"
        assert isinstance(data["requests"], list), f"{metric_name} requests is not a list"

    return test_impl


def make_compute_endpoint_error_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    request_payload: Dict[str, Any],
    expected_status_code: int,
    expected_error_substring: str,
    setup_mocks: bool = True,
    df_type: Literal["Pandas", "Polars"] = "Polars",
) -> Callable[[], None]:
    """
    Factory to create a test for compute endpoint error cases.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path
    :param client: TestClient instance for making requests
    :param request_payload: Request payload that should trigger an error
    :param expected_status_code: Expected HTTP error status code
    :param expected_error_substring: Substring expected in error message
    :param setup_mocks: Whether to setup data source mocks (False for validation errors)
    :return: Test function
    """

    @patch(f"{module_path}.get_data_source")
    def test_impl(self: Any, mock_ds: MagicMock) -> None:
        """Test compute endpoint error handling."""
        if setup_mocks:
            # Create sample dataframe
            sample_df = _create_sample_dataframe(["feature1", "feature2"], df_type=df_type)

            # Mock data source
            mock_data_source = MagicMock()
            mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=sample_df)
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

        # Send request
        response = client.post(endpoint_path, json=request_payload)

        # Verify error response
        assert response.status_code == expected_status_code, (
            f"{metric_name} should return {expected_status_code}, got {response.status_code}: {response.text}"
        )

        data = response.json()
        assert "detail" in data, f"Missing 'detail' in {metric_name} error response"

        # Handle both string detail (controlled errors) and list detail (422 validation errors)
        detail = data["detail"]
        if isinstance(detail, list):
            # For 422 validation errors, concatenate all error messages
            detail_text = " ".join(str(err.get("msg", err)) if isinstance(err, dict) else str(err) for err in detail)
        else:
            detail_text = str(detail)

        assert expected_error_substring.lower() in detail_text.lower(), (
            f"Expected error containing '{expected_error_substring}' in {metric_name}, got: {detail}"
        )

    return test_impl


def make_schedule_endpoint_error_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    request_payload: Dict[str, Any],
    expected_status_code: int,
    expected_error_substring: str,
    mock_scheduler_none: bool = False,
    register_side_effect: Exception | None = None,
) -> Callable[[], None]:
    """
    Factory to create a test for schedule endpoint error cases.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path
    :param client: TestClient instance for making requests
    :param request_payload: Request payload
    :param expected_status_code: Expected HTTP error status code
    :param expected_error_substring: Substring expected in error message
    :param mock_scheduler_none: If True, mock scheduler as None to test unavailability
    :param register_side_effect: Exception to raise from scheduler.register() (e.g., connection errors)
    :return: Test function
    """

    @patch(f"{module_path}.get_prometheus_scheduler")
    @patch(f"{module_path}.get_data_source")
    def test_impl(self: Any, mock_ds: MagicMock, mock_sched_fn: MagicMock) -> None:
        """Test schedule endpoint error handling."""
        if mock_scheduler_none:
            # Mock scheduler as unavailable
            mock_sched_fn.return_value = None
        else:
            # Mock normal scheduler
            mock_sched = MagicMock()
            if register_side_effect:
                # Mock scheduler.register() to raise an exception
                mock_sched.register = AsyncMock(side_effect=register_side_effect)
            else:
                mock_sched.register = AsyncMock(return_value=None)
            mock_sched_fn.return_value = mock_sched

        # Mock data source
        mock_data_source = MagicMock()
        mock_data_source.get_metadata = AsyncMock(return_value={"feature1": "type1"})
        mock_ds.return_value = mock_data_source

        # Send request
        response = client.post(endpoint_path, json=request_payload)

        # Verify error response
        assert response.status_code == expected_status_code, (
            f"{metric_name} should return {expected_status_code}, got {response.status_code}: {response.text}"
        )

        data = response.json()
        assert "detail" in data, f"Missing 'detail' in {metric_name} error response"

        # Handle both string detail (controlled errors) and list detail (422 validation errors)
        detail = data["detail"]
        if isinstance(detail, list):
            # For 422 validation errors, concatenate all error messages
            detail_text = " ".join(str(err.get("msg", err)) if isinstance(err, dict) else str(err) for err in detail)
        else:
            detail_text = str(detail)

        assert expected_error_substring.lower() in detail_text.lower(), (
            f"Expected error containing '{expected_error_substring}' in {metric_name}, got: {detail}"
        )

    return test_impl


def make_delete_endpoint_error_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    request_id: str,
    expected_status_code: int,
    expected_error_substring: str,
    mock_scheduler_none: bool = False,
    delete_side_effect: Exception | None = None,
) -> Callable[[], None]:
    """
    Factory to create a test for delete endpoint error cases.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path
    :param client: TestClient instance for making requests
    :param request_id: Request ID to delete (can be invalid)
    :param expected_status_code: Expected HTTP error status code
    :param expected_error_substring: Substring expected in error message
    :param mock_scheduler_none: If True, mock scheduler as None to test unavailability
    :param delete_side_effect: Exception to raise from scheduler.delete() (e.g., connection errors)
    :return: Test function
    """

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(self: Any, mock_sched_fn: MagicMock) -> None:
        """Test delete endpoint error handling."""
        if mock_scheduler_none:
            # Mock scheduler as unavailable
            mock_sched_fn.return_value = None
        else:
            # Mock normal scheduler
            mock_sched = MagicMock()
            if delete_side_effect:
                # Mock scheduler.delete() to raise an exception
                mock_sched.delete = AsyncMock(side_effect=delete_side_effect)
            else:
                mock_sched.delete = AsyncMock(return_value=None)
            mock_sched_fn.return_value = mock_sched

        # Send request
        payload = {"requestId": request_id}
        response = client.request("DELETE", endpoint_path, json=payload)

        # Verify error response
        assert response.status_code == expected_status_code, (
            f"{metric_name} should return {expected_status_code}, got {response.status_code}: {response.text}"
        )

        data = response.json()
        assert "detail" in data, f"Missing 'detail' in {metric_name} error response"

        # Handle both string detail (controlled errors) and list detail (422 validation errors)
        detail = data["detail"]
        if isinstance(detail, list):
            # For 422 validation errors, concatenate all error messages
            detail_text = " ".join(str(err.get("msg", err)) if isinstance(err, dict) else str(err) for err in detail)
        else:
            detail_text = str(detail)

        assert expected_error_substring.lower() in detail_text.lower(), (
            f"Expected error containing '{expected_error_substring}' in {metric_name}, got: {detail}"
        )

    return test_impl


def make_list_requests_with_data_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    num_requests: int = 2,
) -> Callable[[], None]:
    """
    Factory to create a test for list endpoint with actual requests.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path
    :param client: TestClient instance for making requests
    :param num_requests: Number of mock requests to return
    :return: Test function
    """

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(self: Any, mock_sched_fn: MagicMock) -> None:
        """Test list endpoint returns multiple requests."""
        # Create mock request objects
        from uuid import uuid4

        mock_requests = {}
        for i in range(num_requests):
            request_id = uuid4()
            mock_request = MagicMock()
            mock_request.model_id = f"test-model-{i}"
            mock_request.batch_size = 100 + i * 10
            mock_request.reference_tag = f"baseline-{i}"
            mock_request.fit_columns = [f"feature{i}", f"feature{i + 1}"]
            mock_requests[request_id] = mock_request

        # Mock scheduler
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        # Send request
        response = client.get(endpoint_path)

        # Verify response
        assert response.status_code == 200, f"{metric_name} list with data failed: {response.text}"
        data = response.json()

        assert "requests" in data, f"Missing 'requests' in {metric_name} list response"
        assert isinstance(data["requests"], list), f"{metric_name} requests is not a list"
        assert len(data["requests"]) == num_requests, (
            f"{metric_name} should return {num_requests} requests, got {len(data['requests'])}"
        )

        # Verify request structure
        for req in data["requests"]:
            assert "requestId" in req, "Missing 'requestId' in request"
            assert "modelId" in req, "Missing 'modelId' in request"
            assert "metricName" in req, "Missing 'metricName' in request"
            assert "batchSize" in req, "Missing 'batchSize' in request"
            assert "referenceTag" in req, "Missing 'referenceTag' in request"
            assert "fitColumns" in req, "Missing 'fitColumns' in request"

    return test_impl


def make_list_requests_with_malformed_data_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    num_valid_requests: int = 2,
    num_malformed_requests: int = 2,
) -> Callable[[], None]:
    """
    Factory to create a test for list endpoint with mix of valid and malformed requests.

    This tests the defensive logic that skips malformed requests and only returns valid ones.
    Malformed requests are those missing required attributes (model_id, batch_size, etc.).

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path
    :param client: TestClient instance for making requests
    :param num_valid_requests: Number of valid mock requests to create
    :param num_malformed_requests: Number of malformed mock requests to create
    :return: Test function
    """

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(self: Any, mock_sched_fn: MagicMock) -> None:
        """Test list endpoint filters out malformed requests."""
        from uuid import uuid4

        mock_requests = {}

        # Create valid requests
        for i in range(num_valid_requests):
            request_id = uuid4()
            mock_request = MagicMock()
            mock_request.model_id = f"test-model-{i}"
            mock_request.batch_size = 100 + i * 10
            mock_request.reference_tag = f"baseline-{i}"
            mock_request.fit_columns = [f"feature{i}", f"feature{i+1}"]
            mock_requests[request_id] = mock_request

        # Create malformed requests (missing various required attributes)
        malformed_scenarios = [
            {"batch_size": 100, "reference_tag": "baseline", "fit_columns": ["f1"]},  # Missing model_id
            {"model_id": "model", "reference_tag": "baseline", "fit_columns": ["f1"]},  # Missing batch_size
            {"model_id": "model", "batch_size": 100, "fit_columns": ["f1"]},  # Missing reference_tag
            {"model_id": "model", "batch_size": 100, "reference_tag": "baseline"},  # Missing fit_columns
        ]

        for i in range(num_malformed_requests):
            request_id = uuid4()
            mock_request = MagicMock(spec=[])  # Start with no attributes

            # Apply one of the malformed scenarios (cycle through them)
            scenario = malformed_scenarios[i % len(malformed_scenarios)]
            for attr, value in scenario.items():
                setattr(mock_request, attr, value)

            mock_requests[request_id] = mock_request

        # Mock scheduler to return both valid and malformed requests
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        # Send request
        response = client.get(endpoint_path)

        # Verify response - should still be 200 (defensive handling)
        assert response.status_code == 200, (
            f"{metric_name} list should handle malformed requests gracefully, got {response.status_code}: {response.text}"
        )

        data = response.json()

        # Verify only valid requests are returned (malformed ones filtered out)
        assert "requests" in data, f"Missing 'requests' in {metric_name} list response"
        assert isinstance(data["requests"], list), f"{metric_name} requests is not a list"
        assert len(data["requests"]) == num_valid_requests, (
            f"{metric_name} should return {num_valid_requests} valid requests (filtering out {num_malformed_requests} malformed), "
            f"got {len(data['requests'])}"
        )

        # Verify all returned requests have the required structure
        for req in data["requests"]:
            assert "requestId" in req, "Missing 'requestId' in request"
            assert "modelId" in req, "Missing 'modelId' in request"
            assert "metricName" in req, "Missing 'metricName' in request"
            assert "batchSize" in req, "Missing 'batchSize' in request"
            assert "referenceTag" in req, "Missing 'referenceTag' in request"
            assert "fitColumns" in req, "Missing 'fitColumns' in request"

            # Verify values are from valid requests
            assert req["modelId"].startswith("test-model-"), "Malformed request was not filtered"

    return test_impl


<<<<<<< HEAD
def make_compute_empty_reference_data_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    request_payload: Dict[str, Any],
) -> Callable[[], None]:
    """
    Factory to create a test for compute endpoint when reference data is empty.

    Tests the 404 error case when no reference data is found for the given tag.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path
    :param client: TestClient instance for making requests
    :param request_payload: Request payload (must include referenceTag)
    :return: Test function
    """

    @patch(f"{module_path}.get_data_source")
    def test_impl(self: Any, mock_ds: MagicMock) -> None:
        """Test compute endpoint with empty reference data."""
        # Mock data source to return empty reference dataframe
        mock_data_source = MagicMock()

        # Return empty DataFrame for reference data
        import polars as pl

        empty_df = pl.DataFrame()
        mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=empty_df)

        # Return non-empty DataFrame for current data (won't be reached)
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=pl.DataFrame({"feature1": [1.0, 2.0, 3.0]}))

        mock_ds.return_value = mock_data_source

        # Send request
        response = client.post(endpoint_path, json=request_payload)

        # Verify 404 error response
        assert response.status_code == 404, f"{metric_name} should return 404 for empty reference data"

        data = response.json()
        assert "detail" in data, f"Missing 'detail' in {metric_name} error response"
        assert "no reference data found" in data["detail"].lower(), f"Expected 'no reference data found' in error"

    return test_impl


def make_compute_empty_current_data_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    request_payload: Dict[str, Any],
) -> Callable[[], None]:
    """
    Factory to create a test for compute endpoint when current data is empty.

    Tests the 404 error case when no current/organic data is found.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path
    :param client: TestClient instance for making requests
    :param request_payload: Request payload (must include referenceTag)
    :return: Test function
    """

    @patch(f"{module_path}.get_data_source")
    def test_impl(self: Any, mock_ds: MagicMock) -> None:
        """Test compute endpoint with empty current data."""
        # Mock data source
        mock_data_source = MagicMock()

        # Return non-empty DataFrame for reference data
        import polars as pl

        reference_df = pl.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=reference_df)

        # Return empty DataFrame for current data
        empty_df = pl.DataFrame()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=empty_df)

        mock_ds.return_value = mock_data_source

        # Send request
        response = client.post(endpoint_path, json=request_payload)

        # Verify 404 error response
        assert response.status_code == 404, f"{metric_name} should return 404 for empty current data"

        data = response.json()
        assert "detail" in data, f"Missing 'detail' in {metric_name} error response"
        assert "no current data found" in data["detail"].lower(), f"Expected 'no current data found' in error"

    return test_impl


def make_list_endpoint_scheduler_unavailable_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
) -> Callable[[], None]:
    """
    Factory to create a test for list endpoint when scheduler is unavailable.

    Tests the 500 error case when scheduler is None in list endpoint.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path (must be the list endpoint)
    :param client: TestClient instance for making requests
    :return: Test function
    """

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(self: Any, mock_sched_fn: MagicMock) -> None:
        """Test list endpoint when scheduler is unavailable."""
        # Mock scheduler as None
        mock_sched_fn.return_value = None

        # Send request
        response = client.get(endpoint_path)

        # Verify 500 error response
        assert response.status_code == 500, f"{metric_name} list should return 500 when scheduler unavailable"

        data = response.json()
        assert "detail" in data, f"Missing 'detail' in {metric_name} error response"
        assert "scheduler not available" in data["detail"].lower(), f"Expected 'scheduler not available' in error"

    return test_impl


def make_list_endpoint_exception_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
) -> Callable[[], None]:
    """
    Factory to create a test for list endpoint generic exception handling.

    Tests the catch-all exception handler in list endpoint.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path (must be the list endpoint)
    :param client: TestClient instance for making requests
    :return: Test function
    """

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(self: Any, mock_sched_fn: MagicMock) -> None:
        """Test list endpoint generic exception handling."""
        # Mock scheduler to raise an exception
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(side_effect=Exception("Database connection failed"))
        mock_sched_fn.return_value = mock_sched

        # Send request
        response = client.get(endpoint_path)

        # Verify 500 error response
        assert response.status_code == 500, f"{metric_name} list should return 500 on exception"

        data = response.json()
        assert "detail" in data, f"Missing 'detail' in {metric_name} error response"
        assert "error listing requests" in data["detail"].lower(), f"Expected 'error listing requests' in error"

    return test_impl


def make_compute_generic_exception_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: Any,
    request_payload: Dict[str, Any],
) -> Callable[[], None]:
    """
    Factory to create a test for compute endpoint generic exception handling.

    Tests the catch-all exception handler that catches unexpected errors.

    :param metric_name: Name of the metric for logging
    :param module_path: Module path for patching
    :param endpoint_path: API endpoint path
    :param client: TestClient instance for making requests
    :param request_payload: Request payload
    :return: Test function
    """

    @patch(f"{module_path}.get_data_source")
    def test_impl(self: Any, mock_ds: MagicMock) -> None:
        """Test compute endpoint generic exception handling."""
        # Mock data source to raise an unexpected exception
        mock_data_source = MagicMock()
        mock_data_source.get_dataframe_by_tag = AsyncMock(side_effect=RuntimeError("Unexpected database error"))

        mock_ds.return_value = mock_data_source

        # Send request
        response = client.post(endpoint_path, json=request_payload)

        # Verify 500 error response
        assert response.status_code == 500, f"{metric_name} should return 500 on unexpected exception"

        data = response.json()
        assert "detail" in data, f"Missing 'detail' in {metric_name} error response"
        assert "error computing metric" in data["detail"].lower(), f"Expected 'error computing metric' in error"

    return test_impl


# ============================================================================
# Helper Functions
# ============================================================================


def _create_sample_dataframe(
    columns: List[str], n_samples: int = 100, df_type: Literal["Pandas", "Polars"] = "Polars"
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Create a sample DataFrame for testing.

    :param columns: List of column names
    :param n_samples: Number of samples to generate
    :param df_type: Type of DataFrame to create ("Pandas" or "Polars")
    :return: Pandas or Polars DataFrame
    """
    data = {}
    for col in columns:
        data[col] = np.random.randn(n_samples)

    if df_type == "Polars":
        return pl.DataFrame(data)
    else:
        return pd.DataFrame(data)
