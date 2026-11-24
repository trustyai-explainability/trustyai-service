"""Tests for GET /metrics/all/requests (all scheduled metric computations)."""

import uuid
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.endpoints import routes
from trustyai_service.endpoints.metrics.drift.compare_means import (
    CompareMeansMetricRequest,
)
from trustyai_service.endpoints.metrics.fairness.group.spd import GroupMetricRequest
from trustyai_service.endpoints.metrics.metrics_info import router

MODULE_PATH = "trustyai_service.endpoints.metrics.metrics_info"

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def _scheduled_requests() -> dict[str, dict[uuid.UUID, object]]:
    """Build one CompareMeans, one MMD and one SPD scheduled request."""
    compare_means = CompareMeansMetricRequest.model_validate(
        {"modelId": "model-a", "referenceTag": "baseline", "fitColumns": ["f1"]}
    )
    spd = GroupMetricRequest.model_validate(
        {
            "modelId": "model-b",
            "protectedAttribute": "sex",
            "outcomeName": "approved",
            "privilegedAttribute": 0,
            "unprivilegedAttribute": 1,
            "favorableOutcome": 1,
        }
    )
    spd.metric_name = "SPD"
    return {
        "CompareMeans": {uuid.UUID(int=1): compare_means},
        "SPD": {uuid.UUID(int=2): spd},
    }


class TestGetAllMetricRequests:
    """Listing scheduled requests in one call."""

    def _get(self, url: str) -> object:
        scheduler = MagicMock()
        scheduler.get_all_requests.return_value = _scheduled_requests()
        with patch(f"{MODULE_PATH}.get_prometheus_scheduler", return_value=scheduler):
            return client.get(url)

    def test_returns_all_requests_by_default(self) -> None:
        """The default response wraps every scheduled request."""
        response = self._get(routes.METRICS_ALL_REQUESTS)

        assert response.status_code == HTTPStatus.OK
        requests = response.json()["requests"]
        assert [entry["request"]["metricName"] for entry in requests] == [
            "CompareMeans",
            "SPD",
        ]

    def test_entries_are_re_postable(self) -> None:
        """Each entry carries the id and the original camelCase payload."""
        response = self._get(routes.METRICS_ALL_REQUESTS)

        first = response.json()["requests"][0]
        assert first["id"] == str(uuid.UUID(int=1))
        assert first["request"]["modelId"] == "model-a"
        assert first["request"]["referenceTag"] == "baseline"
        assert first["request"]["fitColumns"] == ["f1"]
        # The dumped payload is valid input for the scheduling endpoint again.
        restored = CompareMeansMetricRequest.model_validate(first["request"])
        assert restored.model_dump(exclude_none=True) == (
            CompareMeansMetricRequest.model_validate(
                {
                    "modelId": "model-a",
                    "referenceTag": "baseline",
                    "fitColumns": ["f1"],
                }
            ).model_dump(exclude_none=True)
        )

    @pytest.mark.parametrize(
        ("metric_type", "expected"),
        [
            ("fairness", ["SPD"]),
            ("drift", ["CompareMeans"]),
            ("all", ["CompareMeans", "SPD"]),
        ],
    )
    def test_type_filter(self, metric_type: str, expected: list[str]) -> None:
        """The type filter selects a metric group; 'all' is the default."""
        response = self._get(f"{routes.METRICS_ALL_REQUESTS}?type={metric_type}")

        assert response.status_code == HTTPStatus.OK
        assert [
            entry["request"]["metricName"] for entry in response.json()["requests"]
        ] == expected

    def test_type_filter_is_case_insensitive(self) -> None:
        """A caller may send FAIRNESS."""
        response = self._get(f"{routes.METRICS_ALL_REQUESTS}?type=FAIRNESS")

        assert response.status_code == HTTPStatus.OK
        assert len(response.json()["requests"]) == 1

    def test_unknown_type_is_rejected(self) -> None:
        """An unknown type is a client error, matching the Java service."""
        response = self._get(f"{routes.METRICS_ALL_REQUESTS}?type=explainers")

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "Invalid metric requests type: explainers" in response.json()["detail"]

    def test_empty_scheduler_returns_an_empty_list(self) -> None:
        """No scheduled metrics is an empty list, not an error."""
        scheduler = MagicMock()
        scheduler.get_all_requests.return_value = {}
        with patch(f"{MODULE_PATH}.get_prometheus_scheduler", return_value=scheduler):
            response = client.get(routes.METRICS_ALL_REQUESTS)

        assert response.status_code == HTTPStatus.OK
        assert response.json() == {"requests": []}

    def test_missing_scheduler_returns_503(self) -> None:
        """Without the scheduler the listing cannot be answered."""
        with patch(f"{MODULE_PATH}.get_prometheus_scheduler", return_value=None):
            response = client.get(routes.METRICS_ALL_REQUESTS)

        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE

    def test_scheduler_error_returns_500(self) -> None:
        """An unexpected scheduler failure is not leaked as a 200."""
        scheduler = MagicMock()
        scheduler.get_all_requests.side_effect = RuntimeError("boom")
        with patch(f"{MODULE_PATH}.get_prometheus_scheduler", return_value=scheduler):
            response = client.get(routes.METRICS_ALL_REQUESTS)

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

    def test_malformed_request_is_skipped(self) -> None:
        """A non-BaseMetricRequest entry does not break the listing."""
        scheduler = MagicMock()
        scheduler.get_all_requests.return_value = {
            "CompareMeans": {uuid.UUID(int=1): object()}
        }
        with patch(f"{MODULE_PATH}.get_prometheus_scheduler", return_value=scheduler):
            response = client.get(routes.METRICS_ALL_REQUESTS)

        assert response.status_code == HTTPStatus.OK
        assert response.json() == {"requests": []}
