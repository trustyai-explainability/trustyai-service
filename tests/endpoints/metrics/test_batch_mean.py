"""Integration tests for the Batch Mean metric endpoints."""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.endpoints.metrics.batch_mean import BatchMeanRequest, router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

MODULE = "trustyai_service.endpoints.metrics.batch_mean"


def _make_dataframe(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible test dataframe with numeric columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature1": rng.standard_normal(n),
            "feature2": rng.uniform(0, 10, n),
            "output": rng.integers(0, 2, n).astype(float),
        }
    )


def _base_payload(**overrides: str | float) -> dict:
    """Create a base request payload with optional overrides."""
    payload: dict[str, str | float] = {
        "modelId": "test-model",
        "columnName": "feature1",
        "batchSize": 100,
    }
    payload.update(overrides)
    return payload


class TestBatchMeanCompute:
    """Tests for the POST /metrics/batchmean endpoint."""

    @patch(f"{MODULE}.get_data_source")
    def test_compute_returns_valid_response(self, mock_ds: MagicMock) -> None:
        """Valid request returns metric name, type, value, and definition."""
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(
            return_value=_make_dataframe()
        )
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/batchmean", json=_base_payload())

        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert data["name"] == "BatchMean"
        assert data["type"] == "BATCH_MEAN"
        assert isinstance(data["value"], float)
        assert "specificDefinition" in data

    @patch(f"{MODULE}.get_data_source")
    def test_compute_correct_mean(self, mock_ds: MagicMock) -> None:
        """Mean of [2, 4, 6, 8, 10] is 6.0."""
        df = pd.DataFrame({"col": [2.0, 4.0, 6.0, 8.0, 10.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/batchmean", json=_base_payload(columnName="col")
        )

        assert response.status_code == HTTPStatus.OK
        assert response.json()["value"] == 6.0  # noqa: PLR2004

    @patch(f"{MODULE}.get_data_source")
    def test_compute_with_thresholds_inside(self, mock_ds: MagicMock) -> None:
        """Value within threshold bounds reports outsideBounds=False."""
        df = pd.DataFrame({"col": [5.0, 5.0, 5.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/batchmean",
            json=_base_payload(
                columnName="col", lowerThreshold=0.0, upperThreshold=10.0
            ),
        )

        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert "thresholds" in data
        assert data["thresholds"]["outsideBounds"] is False

    @patch(f"{MODULE}.get_data_source")
    def test_compute_with_thresholds_outside(self, mock_ds: MagicMock) -> None:
        """Value outside threshold bounds reports outsideBounds=True."""
        df = pd.DataFrame({"col": [100.0, 100.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/batchmean",
            json=_base_payload(
                columnName="col", lowerThreshold=0.0, upperThreshold=10.0
            ),
        )

        assert response.status_code == HTTPStatus.OK
        assert response.json()["thresholds"]["outsideBounds"] is True

    @patch(f"{MODULE}.get_data_source")
    def test_compute_no_thresholds_omits_field(self, mock_ds: MagicMock) -> None:
        """Response omits thresholds when none are specified."""
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(
            return_value=_make_dataframe()
        )
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/batchmean", json=_base_payload())

        assert response.status_code == HTTPStatus.OK
        assert "thresholds" not in response.json()

    @patch(f"{MODULE}.get_data_source")
    def test_compute_empty_data_returns_404(self, mock_ds: MagicMock) -> None:
        """Empty dataframe returns 404."""
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=pd.DataFrame())
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/batchmean", json=_base_payload())
        assert response.status_code == HTTPStatus.NOT_FOUND

    @patch(f"{MODULE}.get_data_source")
    def test_compute_missing_column_raises(self, mock_ds: MagicMock) -> None:
        """Requesting a non-existent column raises ValueError."""
        df = pd.DataFrame({"other_col": [1.0, 2.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        with pytest.raises(ValueError, match="not found"):
            client.post(
                "/metrics/batchmean",
                json=_base_payload(columnName="nonexistent"),
            )


class TestBatchMeanDefinition:
    """Tests for the definition and interpretation endpoints."""

    def test_definition_returns_name_and_description(self) -> None:
        """GET definition returns name and non-empty description."""
        response = client.get("/metrics/batchmean/definition")
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert "Batch Mean" in data["name"]
        assert len(data["description"]) > 0

    def test_interpret_value(self) -> None:
        """POST definition returns an interpretation string."""
        response = client.post(
            "/metrics/batchmean/definition",
            json=_base_payload(),
        )
        assert response.status_code == HTTPStatus.OK
        assert "interpretation" in response.json()


class TestBatchMeanSchedule:
    """Tests for scheduling and deleting metric computations."""

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_schedule_returns_request_id(self, mock_sched_fn: MagicMock) -> None:
        """Scheduling returns a UUID request ID."""
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.post("/metrics/batchmean/request", json=_base_payload())

        assert response.status_code == HTTPStatus.OK
        assert "requestId" in response.json()
        assert "-" in response.json()["requestId"]

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_delete_schedule(self, mock_sched_fn: MagicMock) -> None:
        """Deleting a valid schedule returns success."""
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            "/metrics/batchmean/request",
            json={"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        )
        assert response.status_code == HTTPStatus.OK
        assert response.json()["status"] == "success"

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_delete_invalid_uuid(self, mock_sched_fn: MagicMock) -> None:
        """Deleting with an invalid UUID returns 400."""
        mock_sched = MagicMock()
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            "/metrics/batchmean/request",
            json={"requestId": "not-a-uuid"},
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST


class TestBatchMeanList:
    """Tests for listing scheduled metric computations."""

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_empty(self, mock_sched_fn: MagicMock) -> None:
        """Empty schedule returns empty list."""
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/batchmean/requests")
        assert response.status_code == HTTPStatus.OK
        assert response.json()["requests"] == []

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_with_requests(self, mock_sched_fn: MagicMock) -> None:
        """Listing returns all scheduled requests with correct fields."""
        num_requests = 3
        mock_requests = {}
        for i in range(num_requests):
            req = MagicMock()
            req.model_id = f"model-{i}"
            req.column_name = "feature1"
            req.batch_size = 100
            req.lower_threshold = None
            req.upper_threshold = None
            mock_requests[uuid4()] = req
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/batchmean/requests")
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert len(data["requests"]) == num_requests
        for req in data["requests"]:
            assert req["metricName"] == "BatchMean"
            assert "columnName" in req

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_filters_malformed(self, mock_sched_fn: MagicMock) -> None:
        """Malformed requests are skipped, valid ones returned."""
        mock_requests = {}
        valid = MagicMock()
        valid.model_id = "m1"
        valid.column_name = "c1"
        valid.batch_size = 100
        valid.lower_threshold = None
        valid.upper_threshold = None
        mock_requests[uuid4()] = valid

        malformed = MagicMock(spec=[])
        mock_requests[uuid4()] = malformed

        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/batchmean/requests")
        assert response.status_code == HTTPStatus.OK
        assert len(response.json()["requests"]) == 1

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_exception_propagates(self, mock_sched_fn: MagicMock) -> None:
        """Unhandled scheduler errors propagate as exceptions."""
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(side_effect=RuntimeError("DB error"))
        mock_sched_fn.return_value = mock_sched

        with pytest.raises(RuntimeError, match="DB error"):
            client.get("/metrics/batchmean/requests")


class TestDeprecatedIdentityEndpoints:
    """Tests for backward-compatible /metrics/identity/* endpoints."""

    @patch(f"{MODULE}.get_data_source")
    def test_deprecated_compute(self, mock_ds: MagicMock) -> None:
        """Deprecated compute forwards to BatchMean and returns correct value."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/identity", json=_base_payload())
        assert response.status_code == HTTPStatus.OK
        assert response.json()["name"] == "BatchMean"
        assert response.json()["value"] == 2.0  # noqa: PLR2004

    def test_deprecated_definition(self) -> None:
        """Deprecated definition forwards to BatchMean definition."""
        response = client.get("/metrics/identity/definition")
        assert response.status_code == HTTPStatus.OK
        assert "Batch Mean" in response.json()["name"]

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_deprecated_schedule(self, mock_sched_fn: MagicMock) -> None:
        """Deprecated schedule forwards to BatchMean schedule."""
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.post("/metrics/identity/request", json=_base_payload())
        assert response.status_code == HTTPStatus.OK
        assert "requestId" in response.json()

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_deprecated_delete(self, mock_sched_fn: MagicMock) -> None:
        """Deprecated delete forwards to BatchMean delete."""
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            "/metrics/identity/request",
            json={"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        )
        assert response.status_code == HTTPStatus.OK

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_deprecated_list(self, mock_sched_fn: MagicMock) -> None:
        """Deprecated list forwards to BatchMean list."""
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/identity/requests")
        assert response.status_code == HTTPStatus.OK
        assert "requests" in response.json()


class TestBatchMeanRequest:
    """Tests for BatchMeanRequest model validation."""

    def test_model_validator_sets_metric_name(self) -> None:
        """Default metric_name is set to BatchMean by model validator."""
        request = BatchMeanRequest(modelId="test", columnName="col")
        assert request.metric_name == "BatchMean"

    def test_retrieve_tags(self) -> None:
        """Tags include columnName and modelId."""
        request = BatchMeanRequest(modelId="test", columnName="col")
        tags = request.retrieve_tags()
        assert tags["columnName"] == "col"
        assert tags["modelId"] == "test"

    def test_default_batch_size(self) -> None:
        """Default batch_size matches the model field default."""
        request = BatchMeanRequest(modelId="test", columnName="col")
        assert request.batch_size == BatchMeanRequest.model_fields["batch_size"].default
