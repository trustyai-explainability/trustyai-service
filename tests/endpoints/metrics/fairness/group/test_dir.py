"""Tests for Disparate Impact Ratio (DIR) fairness metric endpoint.

Covers all six endpoint routes plus their deprecated aliases:

- POST /metrics/group/fairness/dir            (compute)
- GET  /metrics/group/fairness/dir/definition  (definition)
- POST /metrics/group/fairness/dir/definition  (interpret - 501)
- POST /metrics/group/fairness/dir/request     (schedule)
- DELETE /metrics/group/fairness/dir/request   (delete schedule)
- GET  /metrics/group/fairness/dir/requests    (list schedules)
"""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.fairness.group.dir import (
    DEFAULT_DIR_THRESHOLD_DELTA,
    DIR_FAIRNESS_TARGET,
    router,
)

from . import factory

# ---------------------------------------------------------------------------
# Test app setup
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router)
client = TestClient(app)

# Module path used for all patches
_MODULE = "src.endpoints.metrics.fairness.group.dir"

# Standard request payload
_PAYLOAD = factory._base_request_payload()

# Response keys expected from the compute endpoint
_COMPUTE_KEYS = ["name", "value", "type", "specificDefinition", "thresholds"]


class TestDIREndpoints:
    """Endpoint tests for Disparate Impact Ratio (DIR) metric."""

    # ====================================================================
    # Success cases
    # ====================================================================

    test_compute_endpoint = factory.make_compute_endpoint_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir",
        client=client,
        request_payload=_PAYLOAD,
        expected_response_keys=_COMPUTE_KEYS,
    )

    test_definition_endpoint = factory.make_definition_endpoint_test(
        metric_name="DIR",
        endpoint_path="/metrics/group/fairness/dir/definition",
        client=client,
        expected_name="Disparate Impact Ratio",
    )

    test_interpret_value_not_implemented = (
        factory.make_interpret_value_not_implemented_test(
            metric_name="DIR",
            endpoint_path="/metrics/group/fairness/dir/definition",
            client=client,
        )
    )

    test_schedule_endpoint = factory.make_schedule_endpoint_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/request",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_delete_schedule_endpoint = factory.make_delete_schedule_endpoint_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/request",
        client=client,
    )

    test_list_requests_endpoint = factory.make_list_requests_endpoint_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/requests",
        client=client,
    )

    # ====================================================================
    # Threshold / delta tests
    # ====================================================================

    test_compute_with_query_delta = factory.make_compute_with_query_delta_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir",
        client=client,
        request_payload=_PAYLOAD,
        query_delta=0.3,
        fairness_target=DIR_FAIRNESS_TARGET,
    )

    test_compute_with_body_delta = (
        factory.make_compute_with_threshold_delta_in_body_test(
            metric_name="DIR",
            module_path=_MODULE,
            endpoint_path="/metrics/group/fairness/dir",
            client=client,
            request_payload=_PAYLOAD,
            body_delta=0.15,
            fairness_target=DIR_FAIRNESS_TARGET,
        )
    )

    def test_compute_uses_default_delta(self) -> None:
        """Verify the default threshold delta is applied when none is provided."""
        with patch(f"{_MODULE}.get_data_source") as mock_ds:
            sample_df = factory._create_fairness_dataframe()
            mock_data_source = MagicMock()
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

            response = client.post("/metrics/group/fairness/dir", json=_PAYLOAD)

            assert response.status_code == HTTPStatus.OK
            thresholds = response.json()["thresholds"]
            assert thresholds["lowerBound"] == (
                DIR_FAIRNESS_TARGET - DEFAULT_DIR_THRESHOLD_DELTA
            )
            assert thresholds["upperBound"] == (
                DIR_FAIRNESS_TARGET + DEFAULT_DIR_THRESHOLD_DELTA
            )

    # ====================================================================
    # Edge cases
    # ====================================================================

    test_list_with_data = factory.make_list_requests_with_data_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/requests",
        client=client,
        num_requests=3,
    )

    test_list_filters_malformed = factory.make_list_requests_with_malformed_data_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/requests",
        client=client,
        num_valid_requests=2,
        num_malformed_requests=2,
    )

    def test_compute_returns_correct_metric_name(self) -> None:
        """Verify the response has name='DIR'."""
        with patch(f"{_MODULE}.get_data_source") as mock_ds:
            sample_df = factory._create_fairness_dataframe()
            mock_data_source = MagicMock()
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

            data = client.post("/metrics/group/fairness/dir", json=_PAYLOAD).json()
            assert data["name"] == "DIR"
            assert data["type"] == "metric"

    def test_compute_dir_value_range(self) -> None:
        """DIR should be >= 0 for valid data."""
        with patch(f"{_MODULE}.get_data_source") as mock_ds:
            sample_df = factory._create_fairness_dataframe()
            mock_data_source = MagicMock()
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

            data = client.post("/metrics/group/fairness/dir", json=_PAYLOAD).json()
            assert data["value"] >= 0

    # ====================================================================
    # Error handling
    # ====================================================================

    test_compute_empty_data = factory.make_compute_empty_data_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_compute_generic_exception = factory.make_compute_generic_exception_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_compute_missing_model_id = factory.make_compute_endpoint_validation_error_test(
        metric_name="DIR",
        endpoint_path="/metrics/group/fairness/dir",
        client=client,
        request_payload={
            # missing modelId
            "protectedAttribute": "gender",
            "outcomeName": "income",
            "privilegedAttribute": "male",
            "unprivilegedAttribute": "female",
            "favorableOutcome": 1,
        },
        expected_status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        expected_error_substring="modelId",
    )

    test_compute_missing_protected_attribute = (
        factory.make_compute_endpoint_validation_error_test(
            metric_name="DIR",
            endpoint_path="/metrics/group/fairness/dir",
            client=client,
            request_payload={
                "modelId": "test-model",
                # missing protectedAttribute
                "outcomeName": "income",
                "privilegedAttribute": "male",
                "unprivilegedAttribute": "female",
                "favorableOutcome": 1,
            },
            expected_status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            expected_error_substring="protectedAttribute",
        )
    )

    test_compute_empty_body = factory.make_compute_endpoint_validation_error_test(
        metric_name="DIR",
        endpoint_path="/metrics/group/fairness/dir",
        client=client,
        request_payload={},
        expected_status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        expected_error_substring="Field required",
    )

    test_delete_invalid_uuid = factory.make_delete_invalid_uuid_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/request",
        client=client,
    )

    # ====================================================================
    # Scheduler error tests
    # ====================================================================

    test_schedule_scheduler_unavailable = (
        factory.make_schedule_scheduler_unavailable_test(
            metric_name="DIR",
            module_path=_MODULE,
            endpoint_path="/metrics/group/fairness/dir/request",
            client=client,
            request_payload=_PAYLOAD,
        )
    )

    test_schedule_register_exception = factory.make_schedule_register_exception_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/request",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_delete_scheduler_unavailable = factory.make_delete_scheduler_unavailable_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/request",
        client=client,
    )

    test_delete_exception = factory.make_delete_exception_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/request",
        client=client,
    )

    test_list_scheduler_unavailable = factory.make_list_scheduler_unavailable_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/requests",
        client=client,
    )

    test_list_exception = factory.make_list_exception_test(
        metric_name="DIR",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/dir/requests",
        client=client,
    )

    # ====================================================================
    # Deprecated endpoint tests
    # ====================================================================

    test_deprecated_compute = factory.make_deprecated_compute_test(
        metric_name="DIR",
        module_path=_MODULE,
        deprecated_path="/dir",
        client=client,
        request_payload=_PAYLOAD,
        expected_response_keys=_COMPUTE_KEYS,
    )

    test_deprecated_definition = factory.make_deprecated_definition_test(
        metric_name="DIR",
        deprecated_path="/dir/definition",
        client=client,
        expected_name_substring="Disparate Impact Ratio",
    )

    test_deprecated_interpret = factory.make_deprecated_interpret_test(
        metric_name="DIR",
        deprecated_path="/dir/definition",
        client=client,
    )

    test_deprecated_schedule = factory.make_deprecated_schedule_test(
        metric_name="DIR",
        module_path=_MODULE,
        deprecated_path="/dir/request",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_deprecated_delete = factory.make_deprecated_delete_test(
        metric_name="DIR",
        module_path=_MODULE,
        deprecated_path="/dir/request",
        client=client,
    )

    test_deprecated_list = factory.make_deprecated_list_test(
        metric_name="DIR",
        module_path=_MODULE,
        deprecated_path="/dir/requests",
        client=client,
    )

    # ====================================================================
    # GroupMetricRequest model tests
    # ====================================================================

    test_retrieve_tags = factory.make_retrieve_tags_test()
