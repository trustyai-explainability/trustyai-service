"""Tests for MMD drift detection endpoint."""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.drift.mmd import MMDMetricRequest, router

from . import factory

goodpoints = pytest.importorskip(
    "goodpoints", reason="goodpoints package not installed"
)

app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestMMDEndpoints:
    """Endpoint tests for MMD metric."""

    test_compute_endpoint_pandas = factory.make_compute_endpoint_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "numPermutations": 39,
            "bandwidth": 1.0,
            "kernel": "gauss",
            "alpha": 0.05,
        },
        expected_response_keys=[
            "status",
            "value",
            "drift_detected",
            "p_value",
            "threshold",
            "alpha",
            "fit_columns",
        ],
        df_type="Pandas",
    )

    test_compute_endpoint_polars = factory.make_compute_endpoint_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "numPermutations": 39,
            "bandwidth": 1.0,
            "kernel": "gauss",
            "alpha": 0.05,
        },
        expected_response_keys=[
            "status",
            "value",
            "drift_detected",
            "p_value",
            "threshold",
            "alpha",
            "fit_columns",
        ],
        df_type="Polars",
    )

    test_definition_endpoint = factory.make_definition_endpoint_test(
        metric_name="MMD",
        endpoint_path="/metrics/drift/mmd/definition",
        client=client,
        expected_name="Maximum Mean Discrepancy",
    )

    test_schedule_endpoint = factory.make_schedule_endpoint_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "alpha": 0.05,
        },
    )

    test_delete_schedule_endpoint = factory.make_delete_schedule_endpoint_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/request",
        client=client,
    )

    test_list_requests_endpoint = factory.make_list_requests_endpoint_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/requests",
        client=client,
    )

    # ========================================================================
    # Compute Endpoint Logic Tests
    # ========================================================================

    def test_multivariate_result_structure(self) -> None:
        """MMD returns a single multivariate result, not per-feature results."""
        rng = np.random.default_rng(42)
        sample_df = pd.DataFrame(
            {
                "feature1": rng.standard_normal(100),
                "feature2": rng.standard_normal(100),
                "feature3": rng.standard_normal(100),
            }
        )

        with patch("src.endpoints.metrics.drift.mmd.get_data_source") as mock_ds:
            mock_data_source = MagicMock()
            mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=sample_df)
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

            response = client.post(
                "/metrics/drift/mmd",
                json={
                    "modelId": "test-model",
                    "referenceTag": "baseline",
                    "fitColumns": ["feature1", "feature2", "feature3"],
                },
            )

            assert response.status_code == HTTPStatus.OK
            data = response.json()

            assert "fit_columns" in data
            assert data["fit_columns"] == ["feature1", "feature2", "feature3"]
            assert "feature_results" not in data
            assert isinstance(data["value"], float)
            assert isinstance(data["drift_detected"], bool)
            assert 0 < data["p_value"] <= 1

    def test_drift_detected_with_shifted_data(self) -> None:
        """MMD endpoint detects drift when current data is shifted."""
        rng = np.random.default_rng(77)
        reference_df = pd.DataFrame(
            {
                "f1": rng.standard_normal(200),
                "f2": rng.standard_normal(200),
            }
        )
        current_df = pd.DataFrame(
            {
                "f1": rng.standard_normal(200) + 5.0,
                "f2": rng.standard_normal(200) + 5.0,
            }
        )

        with patch("src.endpoints.metrics.drift.mmd.get_data_source") as mock_ds:
            mock_data_source = MagicMock()
            mock_data_source.get_dataframe_by_tag = AsyncMock(
                return_value=reference_df,
            )
            mock_data_source.get_organic_dataframe = AsyncMock(
                return_value=current_df,
            )
            mock_ds.return_value = mock_data_source

            response = client.post(
                "/metrics/drift/mmd",
                json={
                    "modelId": "test-model",
                    "referenceTag": "baseline",
                    "fitColumns": ["f1", "f2"],
                    "seed": 77,
                },
            )

            assert response.status_code == HTTPStatus.OK
            assert response.json()["drift_detected"] is True

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    test_compute_missing_reference_tag = factory.make_compute_endpoint_error_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd",
        client=client,
        request_payload={
            "modelId": "test-model",
            "fitColumns": ["feature1"],
        },
        expected_status_code=HTTPStatus.BAD_REQUEST,
        expected_error_substring="referenceTag is required",
    )

    test_compute_missing_fit_columns_derives_from_metadata = (
        factory.make_compute_endpoint_test(
            metric_name="MMD",
            module_path="src.endpoints.metrics.drift.mmd",
            endpoint_path="/metrics/drift/mmd",
            client=client,
            request_payload={
                "modelId": "test-model",
                "referenceTag": "baseline",
            },
            expected_response_keys=["status", "value", "drift_detected"],
        )
    )

    test_compute_invalid_feature = factory.make_compute_endpoint_error_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["nonexistent_feature"],
        },
        expected_status_code=HTTPStatus.BAD_REQUEST,
        expected_error_substring="not found in data",
    )

    test_delete_invalid_uuid = factory.make_delete_endpoint_error_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/request",
        client=client,
        request_id="not-a-valid-uuid",
        expected_status_code=HTTPStatus.BAD_REQUEST,
        expected_error_substring="Invalid request ID",
    )

    test_list_requests_with_data = factory.make_list_requests_with_data_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/requests",
        client=client,
        num_requests=3,
    )

    test_list_requests_filters_malformed = (
        factory.make_list_requests_with_malformed_data_test(
            metric_name="MMD",
            module_path="src.endpoints.metrics.drift.mmd",
            endpoint_path="/metrics/drift/mmd/requests",
            client=client,
            num_valid_requests=2,
            num_malformed_requests=2,
        )
    )

    test_compute_empty_reference_data = factory.make_compute_empty_reference_data_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    test_compute_empty_current_data = factory.make_compute_empty_current_data_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
    )

    test_list_scheduler_unavailable = (
        factory.make_list_endpoint_scheduler_unavailable_test(
            metric_name="MMD",
            module_path="src.endpoints.metrics.drift.mmd",
            endpoint_path="/metrics/drift/mmd/requests",
            client=client,
        )
    )

    test_list_exception = factory.make_list_endpoint_exception_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/requests",
        client=client,
    )

    def test_compute_data_fetch_exception(self) -> None:
        """Data-fetch errors return 500 with 'Error fetching data' message."""
        with patch("src.endpoints.metrics.drift.mmd.get_data_source") as mock_ds:
            mock_data_source = MagicMock()
            mock_data_source.get_dataframe_by_tag = AsyncMock(
                side_effect=RuntimeError("Unexpected database error"),
            )
            mock_ds.return_value = mock_data_source

            response = client.post(
                "/metrics/drift/mmd",
                json={
                    "modelId": "test-model",
                    "referenceTag": "baseline",
                    "fitColumns": ["feature1"],
                },
            )

            assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
            assert "error fetching data" in response.json()["detail"].lower()

    def test_compute_goodpoints_import_error(self) -> None:
        """ImportError returns 503 when goodpoints is missing."""
        sample_df = pd.DataFrame(
            {"feature1": np.random.default_rng(0).standard_normal(50)}
        )

        with (
            patch("src.endpoints.metrics.drift.mmd.get_data_source") as mock_ds,
            patch(
                "src.endpoints.metrics.drift.mmd.MMD.compute",
                side_effect=ImportError("goodpoints not installed"),
            ),
        ):
            mock_data_source = MagicMock()
            mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=sample_df)
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

            response = client.post(
                "/metrics/drift/mmd",
                json={
                    "modelId": "test-model",
                    "referenceTag": "baseline",
                    "fitColumns": ["feature1"],
                },
            )

            assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
            assert "goodpoints" in response.json()["detail"].lower()

    def test_compute_whitespace_only_fit_columns(self) -> None:
        """FitColumns with only whitespace should be rejected."""
        response = client.post(
            "/metrics/drift/mmd",
            json={
                "modelId": "test-model",
                "referenceTag": "baseline",
                "fitColumns": ["  ", ""],
            },
        )

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "non-empty" in response.json()["detail"].lower()

    def test_compute_metric_exception(self) -> None:
        """Computation errors return 500 with 'Error computing metric' message."""
        sample_df = pd.DataFrame(
            {"feature1": np.random.default_rng(0).standard_normal(50)}
        )

        with (
            patch("src.endpoints.metrics.drift.mmd.get_data_source") as mock_ds,
            patch(
                "src.endpoints.metrics.drift.mmd.MMD.compute",
                side_effect=RuntimeError("bad compute"),
            ),
        ):
            mock_data_source = MagicMock()
            mock_data_source.get_dataframe_by_tag = AsyncMock(return_value=sample_df)
            mock_data_source.get_organic_dataframe = AsyncMock(return_value=sample_df)
            mock_ds.return_value = mock_data_source

            response = client.post(
                "/metrics/drift/mmd",
                json={
                    "modelId": "test-model",
                    "referenceTag": "baseline",
                    "fitColumns": ["feature1"],
                },
            )

            assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
            assert "error computing metric" in response.json()["detail"].lower()

    # ========================================================================
    # Scheduler Error Tests
    # ========================================================================

    test_schedule_scheduler_unavailable = factory.make_schedule_endpoint_error_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        expected_error_substring="scheduler not available",
        mock_scheduler_none=True,
    )

    test_schedule_register_exception = factory.make_schedule_endpoint_error_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/request",
        client=client,
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
        },
        expected_status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        expected_error_substring="Error scheduling metric",
        register_side_effect=Exception("Database connection failed"),
    )

    test_delete_scheduler_unavailable = factory.make_delete_endpoint_error_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        expected_error_substring="scheduler not available",
        mock_scheduler_none=True,
    )

    test_delete_exception = factory.make_delete_endpoint_error_test(
        metric_name="MMD",
        module_path="src.endpoints.metrics.drift.mmd",
        endpoint_path="/metrics/drift/mmd/request",
        client=client,
        request_id="123e4567-e89b-12d3-a456-426614174000",
        expected_status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        expected_error_substring="Error deleting schedule",
        delete_side_effect=Exception("Database connection failed"),
    )

    # ========================================================================
    # Retrieve Tags Tests
    # ========================================================================

    test_retrieve_tags_with_all_fields = (
        factory.make_retrieve_tags_with_all_fields_test(
            request_class=MMDMetricRequest,
        )
    )

    test_retrieve_tags_without_reference_tag = (
        factory.make_retrieve_tags_without_reference_tag_test(
            request_class=MMDMetricRequest,
        )
    )

    test_retrieve_tags_without_fit_columns = (
        factory.make_retrieve_tags_without_fit_columns_test(
            request_class=MMDMetricRequest,
        )
    )

    test_retrieve_tags_with_empty_fit_columns = (
        factory.make_retrieve_tags_with_empty_fit_columns_test(
            request_class=MMDMetricRequest,
        )
    )

    test_retrieve_default_tags_with_none_metric_name = (
        factory.make_retrieve_default_tags_with_none_metric_name_test(
            request_class=MMDMetricRequest,
            expected_metric_name="MMD",
        )
    )

    test_retrieve_default_tags_called_directly_by_prometheus_publisher = (
        factory.make_retrieve_default_tags_called_directly_by_prometheus_publisher_test(
            request_class=MMDMetricRequest,
            expected_metric_name="MMD",
        )
    )

    # ========================================================================
    # Deprecated FourierMMD Endpoint Tests
    # ========================================================================

    test_deprecated_compute_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="FourierMMD",
        deprecated_endpoint_path="/metrics/drift/fouriermmd",
        client=client,
        endpoint_type="compute",
        module_path="src.endpoints.metrics.drift.mmd",
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1", "feature2"],
            "batchSize": 100,
            "numPermutations": 39,
            "bandwidth": 1.0,
            "kernel": "gauss",
            "alpha": 0.05,
        },
        expected_response_keys=[
            "status",
            "value",
            "drift_detected",
            "p_value",
            "threshold",
            "alpha",
            "fit_columns",
        ],
    )

    test_deprecated_definition_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="FourierMMD",
        deprecated_endpoint_path="/metrics/drift/fouriermmd/definition",
        client=client,
        endpoint_type="definition",
        expected_name_substring="Maximum Mean Discrepancy",
    )

    test_deprecated_schedule_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="FourierMMD",
        deprecated_endpoint_path="/metrics/drift/fouriermmd/request",
        client=client,
        endpoint_type="schedule",
        module_path="src.endpoints.metrics.drift.mmd",
        request_payload={
            "modelId": "test-model",
            "referenceTag": "baseline",
            "fitColumns": ["feature1"],
            "alpha": 0.05,
        },
    )

    test_deprecated_delete_schedule_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="FourierMMD",
        deprecated_endpoint_path="/metrics/drift/fouriermmd/request",
        client=client,
        endpoint_type="delete",
        module_path="src.endpoints.metrics.drift.mmd",
    )

    test_deprecated_list_requests_endpoint = factory.make_deprecated_endpoint_test(
        metric_name="FourierMMD",
        deprecated_endpoint_path="/metrics/drift/fouriermmd/requests",
        client=client,
        endpoint_type="list",
        module_path="src.endpoints.metrics.drift.mmd",
    )
