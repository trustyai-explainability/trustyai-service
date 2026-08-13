"""Unified test factories for group fairness metric endpoints (DIR, SPD).

This module provides factory functions to create common tests that apply to all
group fairness metric endpoints. Follows the same pattern as the drift metric
test factory in ``tests/endpoints/metrics/drift/factory.py``.

All group fairness endpoints share a common structure:

- POST /metrics/group/fairness/{metric}           - Compute metric
- GET  /metrics/group/fairness/{metric}/definition - Get metric definition
- POST /metrics/group/fairness/{metric}/definition - Interpret specific value
- POST /metrics/group/fairness/{metric}/request    - Schedule recurring computation
- DELETE /metrics/group/fairness/{metric}/request  - Delete scheduled computation
- GET  /metrics/group/fairness/{metric}/requests   - List scheduled computations

Common request fields (camelCase, matching the Java API):

- modelId: Model identifier
- protectedAttribute: Name of the demographic feature column
- outcomeName: Name of the outcome/prediction column
- privilegedAttribute: Value(s) identifying the privileged group
- unprivilegedAttribute: Value(s) identifying the unprivileged group
- favorableOutcome: Value(s) considered a positive/favorable outcome
- batchSize: Optional batch size for data retrieval
- thresholdDelta: Optional threshold delta for fairness bounds
"""

from collections.abc import Callable
from http import HTTPStatus
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# ============================================================================
# Sample data helpers
# ============================================================================


def _create_fairness_dataframe(
    protected_attr: str = "gender",
    outcome_name: str = "income",
    n_samples: int = 200,
) -> pd.DataFrame:
    """Create a sample Pandas DataFrame suitable for fairness metric testing.

    Produces a balanced dataset with two demographic groups and binary outcomes
    designed so that the privileged group has a higher favorable-outcome rate,
    making SPD != 0 and DIR != 1.

    Args:
        protected_attr: Column name for the protected attribute.
        outcome_name: Column name for the outcome variable.
        n_samples: Total number of rows (split evenly between groups).

    Returns:
        Pandas DataFrame with ``protected_attr`` and ``outcome_name`` columns.

    """
    rng = np.random.default_rng(42)
    half = n_samples // 2

    # Privileged group: ~80% favorable outcome rate
    priv_outcomes = rng.choice([1, 0], size=half, p=[0.8, 0.2])
    # Unprivileged group: ~50% favorable outcome rate
    unpriv_outcomes = rng.choice([1, 0], size=half, p=[0.5, 0.5])

    return pd.DataFrame(
        {
            protected_attr: ["male"] * half + ["female"] * half,
            outcome_name: np.concatenate([priv_outcomes, unpriv_outcomes]),
        }
    )


# ============================================================================
# Standard request payload
# ============================================================================


def _base_request_payload() -> dict[str, Any]:
    """Return a minimal valid request payload for group fairness endpoints."""
    return {
        "modelId": "test-model",
        "protectedAttribute": "gender",
        "outcomeName": "income",
        "privilegedAttribute": "male",
        "unprivilegedAttribute": "female",
        "favorableOutcome": 1,
        "batchSize": 100,
    }


# ============================================================================
# Success-case factories
# ============================================================================


def make_compute_endpoint_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
    expected_response_keys: list[str],
) -> Callable[[], None]:
    """Create a test that verifies the compute endpoint returns a valid response.

    Args:
        metric_name: Human-readable metric name for assertion messages.
        module_path: Dotted module path for ``unittest.mock.patch``.
        endpoint_path: HTTP path (e.g. ``/metrics/group/fairness/dir``).
        client: ``TestClient`` wired to the metric router.
        request_payload: JSON body to POST.
        expected_response_keys: Top-level keys that must appear in the response.

    Returns:
        A test callable (bound to ``self`` when used as a class attribute).

    """

    @patch(f"{module_path}.get_data_source")
    def test_impl(_: object, mock_ds: MagicMock) -> None:
        sample_df = _create_fairness_dataframe(
            protected_attr=request_payload.get("protectedAttribute", "gender"),
            outcome_name=request_payload.get("outcomeName", "income"),
        )
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
        mock_ds.return_value = mock_data_source

        response = client.post(endpoint_path, json=request_payload)

        assert response.status_code == HTTPStatus.OK, (
            f"{metric_name} compute failed: {response.text}"
        )
        data = response.json()
        for key in expected_response_keys:
            assert key in data, f"Missing key '{key}' in {metric_name} response"
        # value must be numeric
        assert isinstance(data["value"], (int, float))
        # thresholds sub-object must have bounds
        assert "lowerBound" in data["thresholds"]
        assert "upperBound" in data["thresholds"]
        assert "outsideBounds" in data["thresholds"]

    return test_impl


def make_definition_endpoint_test(
    metric_name: str,
    endpoint_path: str,
    client: TestClient,
    expected_name: str,
) -> Callable[[Any], None]:
    """Create a test that verifies the GET definition endpoint."""

    def test_impl(_: object) -> None:
        response = client.get(endpoint_path)

        assert response.status_code == HTTPStatus.OK, (
            f"{metric_name} definition failed: {response.text}"
        )
        data = response.json()
        assert "name" in data
        assert "description" in data
        assert expected_name.lower() in data["name"].lower(), (
            f"Expected '{expected_name}' in name, got: {data['name']}"
        )
        assert len(data["description"]) > 0

    return test_impl


def make_interpret_value_not_implemented_test(
    metric_name: str,
    endpoint_path: str,
    client: TestClient,
) -> Callable[[Any], None]:
    """Create a test that verifies the POST definition (interpret) endpoint returns 501."""

    def test_impl(_: object) -> None:
        payload = {
            "modelId": "test-model",
            "protectedAttribute": "gender",
            "outcomeName": "income",
            "privilegedAttribute": "male",
            "unprivilegedAttribute": "female",
            "favorableOutcome": 1,
            "metricValue": {"value": 0.85},
        }
        response = client.post(endpoint_path, json=payload)

        assert response.status_code == HTTPStatus.NOT_IMPLEMENTED, (
            f"{metric_name} interpret should return 501, got "
            f"{response.status_code}: {response.text}"
        )
        data = response.json()
        assert "detail" in data
        assert "not yet implemented" in data["detail"].lower()

    return test_impl


def make_schedule_endpoint_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
) -> Callable[[], None]:
    """Create a test that verifies the schedule endpoint returns a requestId."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.post(endpoint_path, json=request_payload)

        assert response.status_code == HTTPStatus.OK, (
            f"{metric_name} schedule failed: {response.text}"
        )
        data = response.json()
        assert "requestId" in data
        request_id = data["requestId"]
        assert isinstance(request_id, str)
        assert "-" in request_id  # basic UUID shape check

    return test_impl


def make_delete_schedule_endpoint_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
) -> Callable[[], None]:
    """Create a test that verifies the delete-schedule endpoint."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        test_uuid = "123e4567-e89b-12d3-a456-426614174000"
        response = client.request(
            "DELETE", endpoint_path, json={"requestId": test_uuid}
        )

        assert response.status_code == HTTPStatus.OK, (
            f"{metric_name} delete schedule failed: {response.text}"
        )
        data = response.json()
        assert data["status"] == "success"

    return test_impl


def make_list_requests_endpoint_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
) -> Callable[[], None]:
    """Create a test that verifies the list-requests endpoint (empty case)."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        response = client.get(endpoint_path)

        assert response.status_code == HTTPStatus.OK, (
            f"{metric_name} list requests failed: {response.text}"
        )
        data = response.json()
        assert "requests" in data
        assert isinstance(data["requests"], list)

    return test_impl


# ============================================================================
# Edge-case factories
# ============================================================================


def make_list_requests_with_data_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    num_requests: int = 2,
) -> Callable[[], None]:
    """Create a test that verifies the list endpoint with populated requests."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_requests: dict[Any, Any] = {}
        for i in range(num_requests):
            request_id = uuid4()
            mock_request = MagicMock()
            mock_request.model_id = f"test-model-{i}"
            mock_request.batch_size = 100 + i * 10
            mock_request.protected_attribute = "gender"
            mock_request.outcome_name = "income"
            mock_requests[request_id] = mock_request

        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        response = client.get(endpoint_path)

        assert response.status_code == HTTPStatus.OK, (
            f"{metric_name} list with data failed: {response.text}"
        )
        data = response.json()
        assert len(data["requests"]) == num_requests

        for req in data["requests"]:
            assert "requestId" in req
            assert "modelId" in req
            assert "metricName" in req
            assert "batchSize" in req
            assert "protectedAttribute" in req
            assert "outcomeName" in req

    return test_impl


def make_list_requests_with_malformed_data_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    num_valid_requests: int = 2,
    num_malformed_requests: int = 2,
) -> Callable[[], None]:
    """Create a test that verifies malformed requests are filtered out."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_requests: dict[Any, Any] = {}

        # Valid requests
        for i in range(num_valid_requests):
            request_id = uuid4()
            mock_request = MagicMock()
            mock_request.model_id = f"test-model-{i}"
            mock_request.batch_size = 100
            mock_request.protected_attribute = "gender"
            mock_request.outcome_name = "income"
            mock_requests[request_id] = mock_request

        # Malformed requests: missing required attributes
        malformed_scenarios: list[dict[str, Any]] = [
            {
                "batch_size": 100,
                "protected_attribute": "gender",
                "outcome_name": "income",
            },  # missing model_id
            {
                "model_id": "model",
                "protected_attribute": "gender",
                "outcome_name": "income",
            },  # missing batch_size
            {
                "model_id": "model",
                "batch_size": 100,
                "outcome_name": "income",
            },  # missing protected_attribute
            {
                "model_id": "model",
                "batch_size": 100,
                "protected_attribute": "gender",
            },  # missing outcome_name
        ]

        for i in range(num_malformed_requests):
            request_id = uuid4()
            mock_request = MagicMock(spec=[])  # no attributes
            scenario = malformed_scenarios[i % len(malformed_scenarios)]
            for attr, value in scenario.items():
                setattr(mock_request, attr, value)
            mock_requests[request_id] = mock_request

        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        response = client.get(endpoint_path)

        assert response.status_code == HTTPStatus.OK, (
            f"{metric_name} list should handle malformed requests, "
            f"got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert len(data["requests"]) == num_valid_requests, (
            f"{metric_name} should return {num_valid_requests} valid requests "
            f"(filtering out {num_malformed_requests} malformed), "
            f"got {len(data['requests'])}"
        )

        for req in data["requests"]:
            assert req["modelId"].startswith("test-model-")

    return test_impl


# ============================================================================
# Error-case factories
# ============================================================================


def make_compute_empty_data_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
) -> Callable[[], None]:
    """Create a test for compute endpoint when no data exists for the model."""

    @patch(f"{module_path}.get_data_source")
    def test_impl(_: object, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=pd.DataFrame())
        mock_ds.return_value = mock_data_source

        response = client.post(endpoint_path, json=request_payload)

        assert response.status_code == HTTPStatus.NOT_FOUND, (
            f"{metric_name} should return 404 for empty data, got "
            f"{response.status_code}: {response.text}"
        )
        data = response.json()
        assert "detail" in data
        assert "no data found" in data["detail"].lower()

    return test_impl


def make_compute_generic_exception_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
) -> Callable[[], None]:
    """Create a test for compute endpoint catch-all exception handler."""

    @patch(f"{module_path}.get_data_source")
    def test_impl(_: object, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(
            side_effect=RuntimeError("Unexpected database error"),
        )
        mock_ds.return_value = mock_data_source

        response = client.post(endpoint_path, json=request_payload)

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, (
            f"{metric_name} should return 500 on unexpected exception"
        )
        data = response.json()
        assert "detail" in data

    return test_impl


def make_compute_endpoint_validation_error_test(
    metric_name: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
    expected_status_code: int,
    expected_error_substring: str,
) -> Callable[[Any], None]:
    """Create a test for request validation errors (422)."""

    def test_impl(_: object) -> None:
        response = client.post(endpoint_path, json=request_payload)

        assert response.status_code == expected_status_code, (
            f"{metric_name} should return {expected_status_code}, got "
            f"{response.status_code}: {response.text}"
        )
        data = response.json()
        assert "detail" in data

        detail = data["detail"]
        if isinstance(detail, list):
            # Include both msg and loc for 422 validation errors so
            # assertions can match on field names (e.g. "modelId").
            parts: list[str] = []
            for err in detail:
                if isinstance(err, dict):
                    parts.append(str(err.get("msg", "")))
                    loc = err.get("loc", [])
                    parts.extend(str(segment) for segment in loc)
                else:
                    parts.append(str(err))
            detail_text = " ".join(parts)
        else:
            detail_text = str(detail)

        assert expected_error_substring.lower() in detail_text.lower(), (
            f"Expected '{expected_error_substring}' in {metric_name} error, "
            f"got: {detail}"
        )

    return test_impl


def make_delete_invalid_uuid_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
) -> Callable[[], None]:
    """Create a test for DELETE with an invalid UUID."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            endpoint_path,
            json={"requestId": "not-a-valid-uuid"},
        )

        assert response.status_code == HTTPStatus.BAD_REQUEST, (
            f"{metric_name} delete should return 400 for invalid UUID"
        )
        data = response.json()
        assert "detail" in data
        assert "invalid request id" in data["detail"].lower()

    return test_impl


def make_schedule_scheduler_unavailable_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
) -> Callable[[], None]:
    """Create a test for schedule endpoint when scheduler is None."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.post(endpoint_path, json=request_payload)

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, (
            f"{metric_name} schedule should return 500 when scheduler unavailable"
        )
        data = response.json()
        assert "detail" in data
        assert "scheduler not available" in data["detail"].lower()

    return test_impl


def make_schedule_register_exception_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
) -> Callable[[], None]:
    """Create a test for schedule endpoint when register() raises."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        mock_sched_fn.return_value = mock_sched

        response = client.post(endpoint_path, json=request_payload)

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, (
            f"{metric_name} schedule should return 500 on register exception"
        )
        data = response.json()
        assert "detail" in data
        assert "error scheduling metric" in data["detail"].lower()

    return test_impl


def make_delete_scheduler_unavailable_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
) -> Callable[[], None]:
    """Create a test for DELETE when scheduler is None."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        test_uuid = "123e4567-e89b-12d3-a456-426614174000"
        response = client.request(
            "DELETE", endpoint_path, json={"requestId": test_uuid}
        )

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, (
            f"{metric_name} delete should return 500 when scheduler unavailable"
        )
        data = response.json()
        assert "detail" in data
        assert "scheduler not available" in data["detail"].lower()

    return test_impl


def make_delete_exception_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
) -> Callable[[], None]:
    """Create a test for DELETE when scheduler.delete() raises."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        mock_sched_fn.return_value = mock_sched

        test_uuid = "123e4567-e89b-12d3-a456-426614174000"
        response = client.request(
            "DELETE", endpoint_path, json={"requestId": test_uuid}
        )

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, (
            f"{metric_name} delete should return 500 on exception"
        )
        data = response.json()
        assert "detail" in data
        assert "error deleting schedule" in data["detail"].lower()

    return test_impl


def make_list_scheduler_unavailable_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
) -> Callable[[], None]:
    """Create a test for list endpoint when scheduler is None."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.get(endpoint_path)

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, (
            f"{metric_name} list should return 500 when scheduler unavailable"
        )
        data = response.json()
        assert "detail" in data
        assert "scheduler not available" in data["detail"].lower()

    return test_impl


def make_list_exception_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
) -> Callable[[], None]:
    """Create a test for list endpoint catch-all exception handler."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(
            side_effect=Exception("Database connection failed"),
        )
        mock_sched_fn.return_value = mock_sched

        response = client.get(endpoint_path)

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, (
            f"{metric_name} list should return 500 on exception"
        )
        data = response.json()
        assert "detail" in data
        assert "error listing requests" in data["detail"].lower()

    return test_impl


# ============================================================================
# Deprecated endpoint factories
# ============================================================================


def make_deprecated_compute_test(
    metric_name: str,
    module_path: str,
    deprecated_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
    expected_response_keys: list[str],
) -> Callable[[], None]:
    """Create a test for a deprecated compute endpoint."""

    @patch(f"{module_path}.get_data_source")
    def test_impl(_: object, mock_ds: MagicMock) -> None:
        sample_df = _create_fairness_dataframe(
            protected_attr=request_payload.get("protectedAttribute", "gender"),
            outcome_name=request_payload.get("outcomeName", "income"),
        )
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
        mock_ds.return_value = mock_data_source

        response = client.post(deprecated_path, json=request_payload)

        assert response.status_code == HTTPStatus.OK, (
            f"Deprecated {metric_name} compute failed: {response.text}"
        )
        data = response.json()
        for key in expected_response_keys:
            assert key in data, f"Missing '{key}' in deprecated {metric_name} response"

    return test_impl


def make_deprecated_definition_test(
    metric_name: str,
    deprecated_path: str,
    client: TestClient,
    expected_name_substring: str,
) -> Callable[[Any], None]:
    """Create a test for a deprecated GET definition endpoint."""

    def test_impl(_: object) -> None:
        response = client.get(deprecated_path)

        assert response.status_code == HTTPStatus.OK, (
            f"Deprecated {metric_name} definition failed: {response.text}"
        )
        data = response.json()
        assert "name" in data
        assert "description" in data
        assert expected_name_substring.lower() in data["name"].lower()

    return test_impl


def make_deprecated_interpret_test(
    metric_name: str,
    deprecated_path: str,
    client: TestClient,
) -> Callable[[Any], None]:
    """Create a test for a deprecated POST definition (interpret) endpoint."""

    def test_impl(_: object) -> None:
        payload = {
            "modelId": "test-model",
            "protectedAttribute": "gender",
            "outcomeName": "income",
            "privilegedAttribute": "male",
            "unprivilegedAttribute": "female",
            "favorableOutcome": 1,
            "metricValue": {"value": 0.85},
        }
        response = client.post(deprecated_path, json=payload)
        assert response.status_code == HTTPStatus.NOT_IMPLEMENTED, (
            f"Deprecated {metric_name} interpret should return 501"
        )

    return test_impl


def make_deprecated_schedule_test(
    metric_name: str,
    module_path: str,
    deprecated_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
) -> Callable[[], None]:
    """Create a test for a deprecated schedule endpoint."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.post(deprecated_path, json=request_payload)

        assert response.status_code == HTTPStatus.OK, (
            f"Deprecated {metric_name} schedule failed: {response.text}"
        )
        data = response.json()
        assert "requestId" in data

    return test_impl


def make_deprecated_delete_test(
    metric_name: str,
    module_path: str,
    deprecated_path: str,
    client: TestClient,
) -> Callable[[], None]:
    """Create a test for a deprecated delete-schedule endpoint."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            deprecated_path,
            json={"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        )

        assert response.status_code == HTTPStatus.OK, (
            f"Deprecated {metric_name} delete failed: {response.text}"
        )
        data = response.json()
        assert data["status"] == "success"

    return test_impl


def make_deprecated_list_test(
    metric_name: str,
    module_path: str,
    deprecated_path: str,
    client: TestClient,
) -> Callable[[], None]:
    """Create a test for a deprecated list-requests endpoint."""

    @patch(f"{module_path}.get_prometheus_scheduler")
    def test_impl(_: object, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        response = client.get(deprecated_path)

        assert response.status_code == HTTPStatus.OK, (
            f"Deprecated {metric_name} list failed: {response.text}"
        )
        data = response.json()
        assert "requests" in data
        assert isinstance(data["requests"], list)

    return test_impl


# ============================================================================
# Compute with custom delta / thresholds
# ============================================================================


def make_compute_with_query_delta_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
    query_delta: float,
    fairness_target: float,
) -> Callable[[], None]:
    """Create a test that passes delta as a query parameter."""

    @patch(f"{module_path}.get_data_source")
    def test_impl(_: object, mock_ds: MagicMock) -> None:
        sample_df = _create_fairness_dataframe(
            protected_attr=request_payload.get("protectedAttribute", "gender"),
            outcome_name=request_payload.get("outcomeName", "income"),
        )
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
        mock_ds.return_value = mock_data_source

        response = client.post(
            f"{endpoint_path}?delta={query_delta}",
            json=request_payload,
        )

        assert response.status_code == HTTPStatus.OK, (
            f"{metric_name} compute with query delta failed: {response.text}"
        )
        data = response.json()
        thresholds = data["thresholds"]
        assert thresholds["lowerBound"] == fairness_target - query_delta
        assert thresholds["upperBound"] == fairness_target + query_delta

    return test_impl


def make_compute_with_threshold_delta_in_body_test(
    metric_name: str,
    module_path: str,
    endpoint_path: str,
    client: TestClient,
    request_payload: dict[str, Any],
    body_delta: float,
    fairness_target: float,
) -> Callable[[], None]:
    """Create a test that passes thresholdDelta in the request body."""

    @patch(f"{module_path}.get_data_source")
    def test_impl(_: object, mock_ds: MagicMock) -> None:
        sample_df = _create_fairness_dataframe(
            protected_attr=request_payload.get("protectedAttribute", "gender"),
            outcome_name=request_payload.get("outcomeName", "income"),
        )
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
        mock_ds.return_value = mock_data_source

        payload_with_delta = {**request_payload, "thresholdDelta": body_delta}
        response = client.post(endpoint_path, json=payload_with_delta)

        assert response.status_code == HTTPStatus.OK, (
            f"{metric_name} compute with body delta failed: {response.text}"
        )
        data = response.json()
        thresholds = data["thresholds"]
        assert thresholds["lowerBound"] == fairness_target - body_delta
        assert thresholds["upperBound"] == fairness_target + body_delta

    return test_impl


# ============================================================================
# GroupMetricRequest.retrieve_tags() test
# ============================================================================


def make_retrieve_tags_test() -> Callable[[Any], None]:
    """Create a test for GroupMetricRequest.retrieve_tags()."""

    def test_impl(_: object) -> None:
        from trustyai_service.endpoints.metrics.fairness.group.utils import (
            GroupMetricRequest,
        )

        request = GroupMetricRequest.model_validate(
            {
                "modelId": "test-model",
                "protectedAttribute": "gender",
                "outcomeName": "income",
                "privilegedAttribute": "male",
                "unprivilegedAttribute": "female",
                "favorableOutcome": 1,
            }
        )
        tags = request.retrieve_tags()
        assert tags["modelId"] == "test-model"
        assert tags["protectedAttribute"] == "gender"
        assert tags["outcomeName"] == "income"

    return test_impl
