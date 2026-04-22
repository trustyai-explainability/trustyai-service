from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.fairness.group.apvd import APVDMetricRequest, router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

MODULE = "src.endpoints.metrics.fairness.group.apvd"


def _make_fairness_dataframe(
    n: int = 200,
    *,
    privileged_value: str = "Male",
    unprivileged_value: str = "Female",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a DataFrame with protected attribute, predictions, and ground truth."""
    rng = np.random.RandomState(seed)
    gender = rng.choice([privileged_value, unprivileged_value], n)
    ground_truth = rng.randint(0, 2, n)

    predictions = ground_truth.copy()
    flip_mask = rng.random(n) < 0.2
    predictions[flip_mask] = 1 - predictions[flip_mask]

    return pd.DataFrame({
        "Gender": gender,
        "prediction": predictions,
        "ground_truth": ground_truth,
    })


def _base_request_payload(**overrides) -> dict:
    payload = {
        "modelId": "test-model",
        "protectedAttribute": "Gender",
        "outcomeName": "prediction",
        "labelName": "ground_truth",
        "privilegedAttribute": "Male",
        "unprivilegedAttribute": "Female",
        "favorableOutcome": 1,
        "batchSize": 200,
    }
    payload.update(overrides)
    return payload


class TestAPVDComputeEndpoint:
    @patch(f"{MODULE}.get_data_source")
    def test_compute_returns_valid_response(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=_make_fairness_dataframe())
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/group/fairness/apvd", json=_base_request_payload())

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "APVD"
        assert data["type"] == "FAIRNESS"
        assert isinstance(data["value"], float)
        assert "thresholds" in data
        assert "lowerBound" in data["thresholds"]
        assert "upperBound" in data["thresholds"]
        assert "outsideBounds" in data["thresholds"]

    @patch(f"{MODULE}.get_data_source")
    def test_compute_value_in_valid_range(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=_make_fairness_dataframe())
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/group/fairness/apvd", json=_base_request_payload())

        assert response.status_code == 200
        value = response.json()["value"]
        assert -1.0 <= value <= 1.0

    @patch(f"{MODULE}.get_data_source")
    def test_compute_with_custom_delta(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=_make_fairness_dataframe())
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/group/fairness/apvd?delta=0.05", json=_base_request_payload())

        assert response.status_code == 200
        thresholds = response.json()["thresholds"]
        assert thresholds["lowerBound"] == -0.05
        assert thresholds["upperBound"] == 0.05

    @patch(f"{MODULE}.get_data_source")
    def test_compute_identical_predictions_gives_zero_apvd(self, mock_ds: MagicMock) -> None:
        """When predictions match ground truth for both groups, APVD should be ~0."""
        df = pd.DataFrame({
            "Gender": ["Male"] * 50 + ["Female"] * 50,
            "prediction": [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25,
            "ground_truth": [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25,
        })
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/group/fairness/apvd", json=_base_request_payload())

        assert response.status_code == 200
        assert abs(response.json()["value"]) < 1e-6

    @patch(f"{MODULE}.get_data_source")
    def test_compute_empty_data_returns_404(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=pd.DataFrame())
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/group/fairness/apvd", json=_base_request_payload())
        assert response.status_code == 404

    @patch(f"{MODULE}.get_data_source")
    def test_compute_missing_label_column_returns_500(self, mock_ds: MagicMock) -> None:
        df = pd.DataFrame({
            "Gender": ["Male", "Female"],
            "prediction": [1, 0],
        })
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=df)
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/group/fairness/apvd", json=_base_request_payload())
        assert response.status_code == 500
        assert "ground_truth" in response.json()["detail"].lower()

    @patch(f"{MODULE}.get_data_source")
    def test_compute_generic_exception(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(side_effect=RuntimeError("DB error"))
        mock_ds.return_value = mock_data_source

        response = client.post("/metrics/group/fairness/apvd", json=_base_request_payload())
        assert response.status_code == 500
        assert "error computing metric" in response.json()["detail"].lower()


class TestAPVDDefinitionEndpoint:
    def test_definition_returns_name_and_description(self) -> None:
        response = client.get("/metrics/group/fairness/apvd/definition")
        assert response.status_code == 200
        data = response.json()
        assert "Average Predictive Value Difference" in data["name"]
        assert len(data["description"]) > 0

    def test_interpret_value(self) -> None:
        payload = {
            "modelId": "test-model",
            "protectedAttribute": "Gender",
            "outcomeName": "prediction",
            "privilegedAttribute": "Male",
            "unprivilegedAttribute": "Female",
            "favorableOutcome": 1,
            "metricValue": {"value": 0.1},
        }
        response = client.post("/metrics/group/fairness/apvd/definition", json=payload)
        assert response.status_code == 200
        assert "interpretation" in response.json()


class TestAPVDScheduleEndpoints:
    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_schedule_returns_request_id(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.post("/metrics/group/fairness/apvd/request", json=_base_request_payload())

        assert response.status_code == 200
        data = response.json()
        assert "requestId" in data
        assert "-" in data["requestId"]

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_schedule_unavailable_scheduler(self, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.post("/metrics/group/fairness/apvd/request", json=_base_request_payload())

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
            "/metrics/group/fairness/apvd/request",
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
            "/metrics/group/fairness/apvd/request",
            json={"requestId": "not-a-uuid"},
        )
        assert response.status_code == 400

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_delete_scheduler_unavailable(self, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.request(
            "DELETE",
            "/metrics/group/fairness/apvd/request",
            json={"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        )
        assert response.status_code == 500


class TestAPVDListEndpoints:
    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_empty(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/group/fairness/apvd/requests")

        assert response.status_code == 200
        assert response.json()["requests"] == []

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_with_requests(self, mock_sched_fn: MagicMock) -> None:
        mock_requests = {}
        for i in range(3):
            req = MagicMock()
            req.model_id = f"model-{i}"
            req.batch_size = 100
            req.protected_attribute = "Gender"
            req.outcome_name = "prediction"
            req.label_name = "ground_truth"
            mock_requests[uuid4()] = req
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/group/fairness/apvd/requests")

        assert response.status_code == 200
        data = response.json()
        assert len(data["requests"]) == 3
        for req in data["requests"]:
            assert req["metricName"] == "APVD"
            assert "labelName" in req

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_filters_malformed(self, mock_sched_fn: MagicMock) -> None:
        mock_requests = {}
        valid_req = MagicMock()
        valid_req.model_id = "model-0"
        valid_req.batch_size = 100
        valid_req.protected_attribute = "Gender"
        valid_req.outcome_name = "prediction"
        valid_req.label_name = "ground_truth"
        mock_requests[uuid4()] = valid_req

        malformed_req = MagicMock(spec=[])
        malformed_req.model_id = "model-1"
        mock_requests[uuid4()] = malformed_req

        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value=mock_requests)
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/group/fairness/apvd/requests")

        assert response.status_code == 200
        assert len(response.json()["requests"]) == 1

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_scheduler_unavailable(self, mock_sched_fn: MagicMock) -> None:
        mock_sched_fn.return_value = None

        response = client.get("/metrics/group/fairness/apvd/requests")
        assert response.status_code == 500

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_list_exception(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(side_effect=Exception("DB error"))
        mock_sched_fn.return_value = mock_sched

        response = client.get("/metrics/group/fairness/apvd/requests")
        assert response.status_code == 500


class TestAPVDDeprecatedEndpoints:
    @patch(f"{MODULE}.get_data_source")
    def test_deprecated_compute(self, mock_ds: MagicMock) -> None:
        mock_data_source = MagicMock()
        mock_data_source.get_organic_dataframe = AsyncMock(return_value=_make_fairness_dataframe())
        mock_ds.return_value = mock_data_source

        response = client.post("/apvd", json=_base_request_payload())
        assert response.status_code == 200
        assert response.json()["name"] == "APVD"

    def test_deprecated_definition(self) -> None:
        response = client.get("/apvd/definition")
        assert response.status_code == 200
        assert "Average Predictive Value Difference" in response.json()["name"]

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_deprecated_schedule(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.register = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.post("/apvd/request", json=_base_request_payload())
        assert response.status_code == 200
        assert "requestId" in response.json()

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_deprecated_delete(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.delete = AsyncMock(return_value=None)
        mock_sched_fn.return_value = mock_sched

        response = client.request(
            "DELETE",
            "/apvd/request",
            json={"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        )
        assert response.status_code == 200

    @patch(f"{MODULE}.get_prometheus_scheduler")
    def test_deprecated_list(self, mock_sched_fn: MagicMock) -> None:
        mock_sched = MagicMock()
        mock_sched.get_requests = MagicMock(return_value={})
        mock_sched_fn.return_value = mock_sched

        response = client.get("/apvd/requests")
        assert response.status_code == 200
        assert "requests" in response.json()


class TestAPVDMetricRequest:
    def test_model_validator_sets_metric_name(self) -> None:
        request = APVDMetricRequest(
            modelId="test-model",
            protectedAttribute="Gender",
            outcomeName="prediction",
            labelName="ground_truth",
            privilegedAttribute="Male",
            unprivilegedAttribute="Female",
            favorableOutcome=1,
        )
        assert request.metric_name == "APVD"

    def test_retrieve_tags_includes_label_name(self) -> None:
        request = APVDMetricRequest(
            modelId="test-model",
            protectedAttribute="Gender",
            outcomeName="prediction",
            labelName="ground_truth",
            privilegedAttribute="Male",
            unprivilegedAttribute="Female",
            favorableOutcome=1,
        )
        tags = request.retrieve_tags()
        assert tags["protectedAttribute"] == "Gender"
        assert tags["outcomeName"] == "prediction"
        assert tags["labelName"] == "ground_truth"
