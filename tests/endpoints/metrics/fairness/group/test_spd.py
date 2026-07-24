"""Tests for Statistical Parity Difference (SPD) fairness metric endpoint.

Covers all six endpoint routes plus their deprecated aliases:

- POST /metrics/group/fairness/spd            (compute)
- GET  /metrics/group/fairness/spd/definition  (definition)
- POST /metrics/group/fairness/spd/definition  (interpret - 501)
- POST /metrics/group/fairness/spd/request     (schedule)
- DELETE /metrics/group/fairness/spd/request   (delete schedule)
- GET  /metrics/group/fairness/spd/requests    (list schedules)
"""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.endpoints.metrics.fairness.group.spd import (
    DEFAULT_SPD_THRESHOLD_DELTA,
    SPD_FAIRNESS_TARGET,
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
_MODULE = "trustyai_service.endpoints.metrics.fairness.group.spd"

# Standard request payload
_PAYLOAD = factory._base_request_payload()

# Response keys expected from the compute endpoint
_COMPUTE_KEYS = ["name", "value", "type", "specificDefinition", "thresholds"]


class TestSPDEndpoints:
    """Endpoint tests for Statistical Parity Difference (SPD) metric."""

    # ====================================================================
    # Success cases
    # ====================================================================

    test_compute_endpoint = factory.make_compute_endpoint_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd",
        client=client,
        request_payload=_PAYLOAD,
        expected_response_keys=_COMPUTE_KEYS,
    )

    test_definition_endpoint = factory.make_definition_endpoint_test(
        metric_name="SPD",
        endpoint_path="/metrics/group/fairness/spd/definition",
        client=client,
        expected_name="Statistical Parity Difference",
    )

    test_interpret_value_not_implemented = (
        factory.make_interpret_value_not_implemented_test(
            metric_name="SPD",
            endpoint_path="/metrics/group/fairness/spd/definition",
            client=client,
        )
    )

    test_schedule_endpoint = factory.make_schedule_endpoint_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/request",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_delete_schedule_endpoint = factory.make_delete_schedule_endpoint_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/request",
        client=client,
    )

    test_list_requests_endpoint = factory.make_list_requests_endpoint_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/requests",
        client=client,
    )

    # ====================================================================
    # Threshold / delta tests
    # ====================================================================

    test_compute_with_query_delta = factory.make_compute_with_query_delta_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd",
        client=client,
        request_payload=_PAYLOAD,
        query_delta=0.2,
        fairness_target=SPD_FAIRNESS_TARGET,
    )

    test_compute_with_body_delta = (
        factory.make_compute_with_threshold_delta_in_body_test(
            metric_name="SPD",
            module_path=_MODULE,
            endpoint_path="/metrics/group/fairness/spd",
            client=client,
            request_payload=_PAYLOAD,
            body_delta=0.05,
            fairness_target=SPD_FAIRNESS_TARGET,
        )
    )

    def test_compute_uses_default_delta(self) -> None:
        """Verify the default threshold delta is applied when none is provided."""
        with patch(f"{_MODULE}.get_data_source") as mock_ds:
            sample_df = factory._create_fairness_dataframe()
            mock_data_source = MagicMock()
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

            response = client.post("/metrics/group/fairness/spd", json=_PAYLOAD)

            assert response.status_code == HTTPStatus.OK
            thresholds = response.json()["thresholds"]
            assert thresholds["lowerBound"] == (
                SPD_FAIRNESS_TARGET - DEFAULT_SPD_THRESHOLD_DELTA
            )
            assert thresholds["upperBound"] == (
                SPD_FAIRNESS_TARGET + DEFAULT_SPD_THRESHOLD_DELTA
            )

    # ====================================================================
    # Edge cases
    # ====================================================================

    test_list_with_data = factory.make_list_requests_with_data_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/requests",
        client=client,
        num_requests=3,
    )

    test_list_filters_malformed = factory.make_list_requests_with_malformed_data_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/requests",
        client=client,
        num_valid_requests=2,
        num_malformed_requests=2,
    )

    def test_compute_returns_correct_metric_name(self) -> None:
        """Verify the response has name='SPD'."""
        with patch(f"{_MODULE}.get_data_source") as mock_ds:
            sample_df = factory._create_fairness_dataframe()
            mock_data_source = MagicMock()
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

            data = client.post("/metrics/group/fairness/spd", json=_PAYLOAD).json()
            assert data["name"] == "SPD"
            assert data["type"] == "metric"

    def test_compute_spd_value_range(self) -> None:
        """SPD should be in [-1, 1] for valid data."""
        with patch(f"{_MODULE}.get_data_source") as mock_ds:
            sample_df = factory._create_fairness_dataframe()
            mock_data_source = MagicMock()
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

            data = client.post("/metrics/group/fairness/spd", json=_PAYLOAD).json()
            assert -1 <= data["value"] <= 1

    # ====================================================================
    # Error handling
    # ====================================================================

    test_compute_empty_data = factory.make_compute_empty_data_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_compute_generic_exception = factory.make_compute_generic_exception_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_compute_missing_model_id = factory.make_compute_endpoint_validation_error_test(
        metric_name="SPD",
        endpoint_path="/metrics/group/fairness/spd",
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
            metric_name="SPD",
            endpoint_path="/metrics/group/fairness/spd",
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
        metric_name="SPD",
        endpoint_path="/metrics/group/fairness/spd",
        client=client,
        request_payload={},
        expected_status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        expected_error_substring="Field required",
    )

    test_delete_invalid_uuid = factory.make_delete_invalid_uuid_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/request",
        client=client,
    )

    # ====================================================================
    # Scheduler error tests
    # ====================================================================

    test_schedule_scheduler_unavailable = (
        factory.make_schedule_scheduler_unavailable_test(
            metric_name="SPD",
            module_path=_MODULE,
            endpoint_path="/metrics/group/fairness/spd/request",
            client=client,
            request_payload=_PAYLOAD,
        )
    )

    test_schedule_register_exception = factory.make_schedule_register_exception_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/request",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_delete_scheduler_unavailable = factory.make_delete_scheduler_unavailable_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/request",
        client=client,
    )

    test_delete_exception = factory.make_delete_exception_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/request",
        client=client,
    )

    test_list_scheduler_unavailable = factory.make_list_scheduler_unavailable_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/requests",
        client=client,
    )

    test_list_exception = factory.make_list_exception_test(
        metric_name="SPD",
        module_path=_MODULE,
        endpoint_path="/metrics/group/fairness/spd/requests",
        client=client,
    )

    # ====================================================================
    # Deprecated endpoint tests
    # ====================================================================

    test_deprecated_compute = factory.make_deprecated_compute_test(
        metric_name="SPD",
        module_path=_MODULE,
        deprecated_path="/spd",
        client=client,
        request_payload=_PAYLOAD,
        expected_response_keys=_COMPUTE_KEYS,
    )

    test_deprecated_definition = factory.make_deprecated_definition_test(
        metric_name="SPD",
        deprecated_path="/spd/definition",
        client=client,
        expected_name_substring="Statistical Parity Difference",
    )

    test_deprecated_interpret = factory.make_deprecated_interpret_test(
        metric_name="SPD",
        deprecated_path="/spd/definition",
        client=client,
    )

    test_deprecated_schedule = factory.make_deprecated_schedule_test(
        metric_name="SPD",
        module_path=_MODULE,
        deprecated_path="/spd/request",
        client=client,
        request_payload=_PAYLOAD,
    )

    test_deprecated_delete = factory.make_deprecated_delete_test(
        metric_name="SPD",
        module_path=_MODULE,
        deprecated_path="/spd/request",
        client=client,
    )

    test_deprecated_list = factory.make_deprecated_list_test(
        metric_name="SPD",
        module_path=_MODULE,
        deprecated_path="/spd/requests",
        client=client,
    )

    # ====================================================================
    # GroupMetricRequest model tests
    # ====================================================================

    test_retrieve_tags = factory.make_retrieve_tags_test()
