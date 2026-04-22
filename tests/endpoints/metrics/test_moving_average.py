from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.moving_average import MovingAverageRequest, router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

MODULE = "src.endpoints.metrics.moving_average"


def _make_dataframe(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "feature1": rng.randn(n),
        "feature2": rng.uniform(0, 10, n),
        "output": rng.randint(0, 2, n).astype(float),
    })


def _base_payload(**overrides) -> dict:
    payload = {
        "modelId": "test-model",
        "columnName": "feature1",
        "batchSize": 100,
    }
    payload.update(overrides)
    return payload


class TestMovingAverageCompute:
    @patch(f"{MODULE}.get_data_source")
    def test_compute_returns_valid_response(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=_make_dataframe())
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/movingaverage", json=_base_payload())

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "MovingAverage"
        assert data["type"] == "IDENTITY"
        assert isinstance(data["value"], float)
        assert "specificDefinition" in data

    @patch(f"{MODULE}.get_data_source")
    def test_compute_correct_mean(self, mock_ds: MagicMock) -> None:
        df = pd.DataFrame({"col": [2.0, 4.0, 6.0, 8.0, 10.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/movingaverage", json=_base_payload(columnName="col"))

        assert response.status_code == 200
        assert response.json()["value"] == 6.0

    @patch(f"{MODULE}.get_data_source")
    def test_compute_with_thresholds_inside(self, mock_ds: MagicMock) -> None:
        df = pd.DataFrame({"col": [5.0, 5.0, 5.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/movingaverage",
            json=_base_payload(columnName="col", lowerThreshold=0.0, upperThreshold=10.0),
        )

        assert response.status_code == 200
        data = response.json()
        assert "thresholds" in data
        assert data["thresholds"]["outsideBounds"] is False

    @patch(f"{MODULE}.get_data_source")
    def test_compute_with_thresholds_outside(self, mock_ds: MagicMock) -> None:
        df = pd.DataFrame({"col": [100.0, 100.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/movingaverage",
            json=_base_payload(columnName="col", lowerThreshold=0.0, upperThreshold=10.0),
        )

        assert response.status_code == 200
        assert response.json()["thresholds"]["outsideBounds"] is True

    @patch(f"{MODULE}.get_data_source")
    def test_compute_no_thresholds_omits_field(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=_make_dataframe())
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/movingaverage", json=_base_payload())

        assert response.status_code == 200
        assert "thresholds" not in response.json()

    @patch(f"{MODULE}.get_data_source")
    def test_compute_empty_data_returns_404(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=pd.DataFrame())
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/movingaverage", json=_base_payload())
        assert response.status_code == 404

    @patch(f"{MODULE}.get_data_source")
    def test_compute_missing_column_returns_500(self, mock_ds: MagicMock) -> None:
        df = pd.DataFrame({"other_col": [1.0, 2.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/movingaverage",
            json=_base_payload(columnName="nonexistent"),
        )
        assert response.status_code == 500
        assert "nonexistent" in response.json()["detail"].lower()

    @patch(f"{MODULE}.get_data_source")
    def test_compute_generic_exception(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(side_effect=RuntimeError("DB error"))
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/movingaverage", json=_base_payload())
        assert response.status_code == 500
        assert "error computing metric" in response.json()["detail"].lower()


class TestMovingAverageDefinition:
    def test_definition_returns_name_and_description(self) -> None:
        response = client.get("/metrics/movingaverage/definition")
        assert response.status_code == 200
        data = response.json()
        assert "Moving Average" in data["name"]
        assert len(data["description"]) > 0

    def test_interpret_value(self) -> None:
        response = client.post(
            "/metrics/movingaverage/definition",
            json=_base_payload(),
        )
        assert response.status_code == 200
        assert "interpretation" in response.json()


class TestMovingAverageSchedule:
    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_schedule_returns_request_id(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.post("/metrics/movingaverage/request", json=_base_payload())

        assert response.status_code == 200
        assert "requestId" in response.json()
        assert "-" in response.json()["requestId"]

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_schedule_unavailable(self, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.post("/metrics/movingaverage/request", json=_base_payload())
        assert response.status_code == 500
        assert "scheduler not available" in response.json()["detail"].lower()

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_delete_schedule(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            "/metrics/movingaverage/request",
            json={"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_delete_invalid_uuid(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            "/metrics/movingaverage/request",
            json={"requestId": "not-a-uuid"},
        )
        assert response.status_code == 400

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_delete_scheduler_unavailable(self, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.request(
            "DELETE",
            "/metrics/movingaverage/request",
            json={"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        )
        assert response.status_code == 500


class TestMovingAverageList:
    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_empty(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/movingaverage/requests")
        assert response.status_code == 200
        assert response.json()["requests"] == []

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_with_requests(self, mock_sched_fn: MagicMock) -> None:
        mock_requests = {}
        for i in range(3):
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

        response = client.get("/metrics/movingaverage/requests")
        assert response.status_code == 200
        data = response.json()
        assert len(data["requests"]) == 3
        for req in data["requests"]:
            assert req["metricName"] == "MovingAverage"
            assert "columnName" in req

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_filters_malformed(self, mock_sched_fn: MagicMock) -> None:
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

        response = client.get("/metrics/movingaverage/requests")
        assert response.status_code == 200
        assert len(response.json()["requests"]) == 1

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_scheduler_unavailable(self, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.get("/metrics/movingaverage/requests")
        assert response.status_code == 500

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_exception(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(side_effect=Exception("DB error"))
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/movingaverage/requests")
        assert response.status_code == 500


class TestDeprecatedIdentityEndpoints:
    @patch(f"{MODULE}.get_data_source")
    def test_deprecated_compute(self, mock_ds: MagicMock) -> None:
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/identity", json=_base_payload())
        assert response.status_code == 200
        assert response.json()["name"] == "MovingAverage"
        assert response.json()["value"] == 2.0

    def test_deprecated_definition(self) -> None:
        response = client.get("/metrics/identity/definition")
        assert response.status_code == 200
        assert "Moving Average" in response.json()["name"]

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_deprecated_schedule(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.post("/metrics/identity/request", json=_base_payload())
        assert response.status_code == 200
        assert "requestId" in response.json()

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_deprecated_delete(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            "/metrics/identity/request",
            json={"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        )
        assert response.status_code == 200

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_deprecated_list(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/identity/requests")
        assert response.status_code == 200
        assert "requests" in response.json()


class TestMovingAverageRequest:
    def test_model_validator_sets_metric_name(self) -> None:
        request = MovingAverageRequest(modelId="test", columnName="col")
        assert request.metric_name == "MovingAverage"

    def test_retrieve_tags(self) -> None:
        request = MovingAverageRequest(modelId="test", columnName="col")
        tags = request.retrieve_tags()
        assert tags["columnName"] == "col"
        assert tags["modelId"] == "test"

    def test_default_batch_size(self) -> None:
        request = MovingAverageRequest(modelId="test", columnName="col")
        assert request.batch_size == 100
