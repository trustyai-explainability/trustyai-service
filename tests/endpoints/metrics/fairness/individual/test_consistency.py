from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.fairness.individual.consistency import (
    IndividualConsistencyRequest,
    compute_consistency_score,
    router,
)

app = FastAPI()
app.include_router(router)
client = TestClient(app)

MODULE = "src.endpoints.metrics.fairness.individual.consistency"


def _make_consistent_dataframe(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a DataFrame where all predictions are the same (perfect consistency)."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "feature1": rng.randn(n),
        "feature2": rng.randn(n),
        "prediction": np.ones(n),
    })


def _make_mixed_dataframe(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a DataFrame with varying predictions."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "feature1": rng.randn(n),
        "feature2": rng.randn(n),
        "prediction": rng.randint(0, 2, n).astype(float),
    })


def _base_request_payload(**overrides) -> dict:
    payload = {
        "modelId": "test-model",
        "outcomeName": "prediction",
        "fitColumns": ["feature1", "feature2"],
        "nNeighbors": 5,
        "batchSize": 100,
    }
    payload.update(overrides)
    return payload


class TestConsistencyComputeEndpoint:
    @patch(f"{MODULE}.get_data_source")
    def test_compute_returns_valid_response(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=_make_mixed_dataframe())
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/individual/fairness/consistency",
            json=_base_request_payload(),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "IndividualConsistency"
        assert data["type"] == "FAIRNESS"
        assert isinstance(data["value"], float)
        assert 0.0 <= data["value"] <= 1.0
        assert "thresholds" in data

    @patch(f"{MODULE}.get_data_source")
    def test_compute_perfect_consistency(self, mock_ds: MagicMock) -> None:
        """All identical predictions should yield consistency = 1.0."""
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(
            return_value=_make_consistent_dataframe(),
        )
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/individual/fairness/consistency",
            json=_base_request_payload(),
        )

        assert response.status_code == 200
        assert abs(response.json()["value"] - 1.0) < 1e-6

    @patch(f"{MODULE}.get_data_source")
    def test_compute_imperfect_consistency(self, mock_ds: MagicMock) -> None:
        """Mixed predictions should yield consistency < 1.0."""
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(
            return_value=_make_mixed_dataframe(),
        )
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/individual/fairness/consistency",
            json=_base_request_payload(),
        )

        assert response.status_code == 200
        value = response.json()["value"]
        assert 0.0 <= value < 1.0

    @patch(f"{MODULE}.get_data_source")
    def test_compute_with_custom_delta(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(
            return_value=_make_mixed_dataframe(),
        )
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/individual/fairness/consistency?delta=0.05",
            json=_base_request_payload(),
        )

        assert response.status_code == 200
        thresholds = response.json()["thresholds"]
        assert thresholds["lowerBound"] == 0.95
        assert thresholds["upperBound"] == 1.05

    @patch(f"{MODULE}.get_data_source")
    def test_compute_empty_data_returns_404(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=pd.DataFrame())
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/individual/fairness/consistency",
            json=_base_request_payload(),
        )
        assert response.status_code == 404

    @patch(f"{MODULE}.get_data_source")
    def test_compute_missing_outcome_column(self, mock_ds: MagicMock) -> None:
        df = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/individual/fairness/consistency",
            json=_base_request_payload(),
        )
        assert response.status_code == 500
        assert "prediction" in response.json()["detail"].lower()

    @patch(f"{MODULE}.get_data_source")
    def test_compute_missing_fit_columns(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(
            return_value=_make_mixed_dataframe(),
        )
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/individual/fairness/consistency",
            json=_base_request_payload(fitColumns=[]),
        )
        assert response.status_code == 500
        assert "fitcolumns" in response.json()["detail"].lower()

    @patch(f"{MODULE}.get_data_source")
    def test_compute_nonexistent_fit_column(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(
            return_value=_make_mixed_dataframe(),
        )
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/individual/fairness/consistency",
            json=_base_request_payload(fitColumns=["nonexistent"]),
        )
        assert response.status_code == 500
        assert "not found in data" in response.json()["detail"].lower()

    @patch(f"{MODULE}.get_data_source")
    def test_compute_generic_exception(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(
            side_effect=RuntimeError("DB error"),
        )
        mock_ds.return_value = mock_data_source

        response = client.post(
            "/metrics/individual/fairness/consistency",
            json=_base_request_payload(),
        )
        assert response.status_code == 500
        assert "error computing metric" in response.json()["detail"].lower()


class TestConsistencyDefinitionEndpoint:
    def test_definition_returns_name_and_description(self) -> None:
        response = client.get("/metrics/individual/fairness/consistency/definition")
        assert response.status_code == 200
        data = response.json()
        assert "Individual Consistency" in data["name"]
        assert len(data["description"]) > 0

    def test_interpret_value(self) -> None:
        response = client.post(
            "/metrics/individual/fairness/consistency/definition",
            json={"metricValue": {"value": 0.9}},
        )
        assert response.status_code == 200
        assert "interpretation" in response.json()


class TestConsistencyScheduleEndpoints:
    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_schedule_returns_request_id(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.post(
            "/metrics/individual/fairness/consistency/request",
            json=_base_request_payload(),
        )

        assert response.status_code == 200
        data = response.json()
        assert "requestId" in data
        assert "-" in data["requestId"]

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_schedule_unavailable_scheduler(self, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.post(
            "/metrics/individual/fairness/consistency/request",
            json=_base_request_payload(),
        )

        assert response.status_code == 500
        assert "scheduler not available" in response.json()["detail"].lower()

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_delete_schedule(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        test_uuid = "123e4567-e89b-12d3-a456-426614174000"
        response = client.request(
            "DELETE",
            "/metrics/individual/fairness/consistency/request",
            json={"requestId": test_uuid},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_delete_invalid_uuid(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            "/metrics/individual/fairness/consistency/request",
            json={"requestId": "not-a-uuid"},
        )
        assert response.status_code == 400

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_delete_scheduler_unavailable(self, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.request(
            "DELETE",
            "/metrics/individual/fairness/consistency/request",
            json={"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        )
        assert response.status_code == 500


class TestConsistencyListEndpoints:
    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_empty(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/individual/fairness/consistency/requests")

        assert response.status_code == 200
        assert response.json()["requests"] == []

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_with_requests(self, mock_sched_fn: MagicMock) -> None:
        mock_requests = {}
        for i in range(3):
            req = MagicMock()
            req.model_id = f"model-{i}"
            req.batch_size = 100
            req.outcome_name = "prediction"
            req.n_neighbors = 5
            req.fit_columns = ["feature1", "feature2"]
            mock_requests[uuid4()] = req
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/individual/fairness/consistency/requests")

        assert response.status_code == 200
        data = response.json()
        assert len(data["requests"]) == 3
        for req in data["requests"]:
            assert req["metricName"] == "IndividualConsistency"
            assert "nNeighbors" in req
            assert "fitColumns" in req

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_filters_malformed(self, mock_sched_fn: MagicMock) -> None:
        mock_requests = {}
        valid_req = MagicMock()
        valid_req.model_id = "model-0"
        valid_req.batch_size = 100
        valid_req.outcome_name = "prediction"
        valid_req.n_neighbors = 5
        valid_req.fit_columns = ["feature1"]
        mock_requests[uuid4()] = valid_req

        malformed_req = MagicMock(spec=[])
        malformed_req.model_id = "model-1"
        mock_requests[uuid4()] = malformed_req

        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/individual/fairness/consistency/requests")

        assert response.status_code == 200
        assert len(response.json()["requests"]) == 1

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_scheduler_unavailable(self, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.get("/metrics/individual/fairness/consistency/requests")
        assert response.status_code == 500

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_exception(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(side_effect=Exception("DB error"))
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/individual/fairness/consistency/requests")
        assert response.status_code == 500


class TestConsistencyScoreFunction:
    def test_perfect_consistency(self) -> None:
        rng = np.random.RandomState(42)
        features = rng.randn(50, 3)
        predictions = np.ones(50)

        score = compute_consistency_score(features, predictions, n_neighbors=5)
        assert abs(score - 1.0) < 1e-6

    def test_imperfect_consistency(self) -> None:
        rng = np.random.RandomState(42)
        features = rng.randn(50, 3)
        predictions = rng.randint(0, 2, 50).astype(float)

        score = compute_consistency_score(features, predictions, n_neighbors=5)
        assert 0.0 <= score <= 1.0
        assert score < 1.0

    def test_n_neighbors_clamped_to_samples(self) -> None:
        """n_neighbors > n_samples-1 should be clamped, not error."""
        features = np.array([[1.0], [2.0], [3.0]])
        predictions = np.array([1.0, 1.0, 1.0])

        score = compute_consistency_score(features, predictions, n_neighbors=100)
        assert abs(score - 1.0) < 1e-6

    def test_empty_data_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="empty"):
            compute_consistency_score(np.array([]), np.array([]), n_neighbors=5)

    def test_single_sample_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="at least 2"):
            compute_consistency_score(
                np.array([[1.0]]),
                np.array([1.0]),
                n_neighbors=5,
            )


class TestIndividualConsistencyRequest:
    def test_model_validator_sets_metric_name(self) -> None:
        request = IndividualConsistencyRequest(
            modelId="test-model",
            outcomeName="prediction",
            fitColumns=["f1", "f2"],
        )
        assert request.metric_name == "IndividualConsistency"

    def test_retrieve_tags(self) -> None:
        request = IndividualConsistencyRequest(
            modelId="test-model",
            outcomeName="prediction",
            fitColumns=["f1", "f2"],
            nNeighbors=10,
        )
        tags = request.retrieve_tags()
        assert tags["outcomeName"] == "prediction"
        assert tags["nNeighbors"] == "10"
        assert tags["fitColumns"] == "f1,f2"

    def test_default_n_neighbors(self) -> None:
        request = IndividualConsistencyRequest(
            modelId="test-model",
            outcomeName="prediction",
            fitColumns=["f1"],
        )
        assert request.n_neighbors == 5
