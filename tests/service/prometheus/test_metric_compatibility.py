"""Contract checks for aliases and selectable Prometheus exposition."""

import importlib
import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from trustyai_service.endpoints.metrics.batch_mean import BatchMeanRequest
from trustyai_service.service.prometheus.gauge_config import GaugeConfig
from trustyai_service.service.prometheus.prometheus_publisher import PrometheusPublisher
from trustyai_service.service.prometheus.prometheus_scheduler import PrometheusScheduler


@pytest.mark.parametrize("schema", ["java", "python-legacy"])
@pytest.mark.parametrize(
    ("module_name", "current", "deprecated", "fields"),
    [
        ("batch_mean", "batchmean", "identity", {"columnName": "feature"}),
        ("drift.compare_means", "drift/comparemeans", "drift/meanshift", {}),
        ("drift.mmd", "drift/mmd", "drift/fouriermmd", {}),
        (
            "drift.kolmogorov_smirnov_streaming",
            "drift/ksteststreaming",
            "drift/approxkstest",
            {},
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_alias_schedule_lifecycle(  # noqa: PLR0913 - route and schema contract matrix
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
    module_name: str,
    current: str,
    deprecated: str,
    fields: dict[str, str],
    *,
    reverse: bool,
) -> None:
    """Create, list, publish and delete through both spellings of a route."""
    monkeypatch.setenv("SERVICE_PROMETHEUS_SCHEMA", schema)
    module = importlib.import_module(
        f"trustyai_service.endpoints.metrics.{module_name}"
    )
    registry = CollectorRegistry()
    publisher = PrometheusPublisher(registry)
    scheduler = PrometheusScheduler(publisher=publisher, data_source=AsyncMock())
    monkeypatch.setattr(module, "get_prometheus_scheduler", lambda: scheduler)
    monkeypatch.setattr(
        "trustyai_service.service.prometheus.prometheus_scheduler.RequestReconciler.reconcile",
        AsyncMock(),
    )
    if hasattr(module, "validate_drift_request"):
        monkeypatch.setattr(module, "validate_drift_request", AsyncMock())
    app = FastAPI()
    app.include_router(module.router)
    client = TestClient(app)
    create_route, delete_route = (
        (current, deprecated) if reverse else (deprecated, current)
    )
    response = client.post(
        f"/metrics/{create_route}/request",
        json={
            "modelId": "model",
            "referenceTag": "baseline",
            "fitColumns": ["feature"],
            **fields,
        },
    )
    assert response.status_code == 200, response.text
    request_id = uuid.UUID(response.json()["requestId"])
    for route in (current, deprecated):
        listed = client.get(f"/metrics/{route}/requests")
        assert listed.status_code == 200
        assert listed.json()["requests"][0]["requestId"] == str(request_id)
    request = scheduler.get_all_requests_flat()[request_id]
    publisher.gauge(
        GaugeConfig(
            model_name="model", request_id=request_id, request=request, value=1.0
        )
    )
    assert f'request="{request_id}"' in generate_latest(registry).decode()
    response = client.request(
        "DELETE",
        f"/metrics/{delete_route}/request",
        json={"requestId": str(request_id)},
    )
    assert response.status_code == 200
    assert not scheduler.get_all_requests_flat()
    assert f'request="{request_id}"' not in generate_latest(registry).decode()


@pytest.mark.parametrize(
    ("schema", "metric_name", "family"),
    [("java", "IDENTITY", "identity"), ("python-legacy", "BatchMean", "batchmean")],
)
def test_exposition_schema(
    monkeypatch: pytest.MonkeyPatch, schema: str, metric_name: str, family: str
) -> None:
    """Switch identifiers at exposition without mutating the REST request."""
    monkeypatch.setenv("SERVICE_PROMETHEUS_SCHEMA", schema)
    registry = CollectorRegistry()
    publisher = PrometheusPublisher(registry)
    request = BatchMeanRequest(
        modelId="model", columnName="feature", metricName="Identity"
    )
    publisher.gauge(
        GaugeConfig(
            model_name="model", request_id=uuid.uuid4(), request=request, value=2.0
        )
    )
    scrape = generate_latest(registry).decode()
    assert f"trustyai_{family}" in scrape
    assert f'metricName="{metric_name}"' in scrape
    assert request.metric_name == "Identity"


def test_invalid_exposition_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unknown schema names at publisher creation."""
    monkeypatch.setenv("SERVICE_PROMETHEUS_SCHEMA", "typo")
    with pytest.raises(ValueError, match="SERVICE_PROMETHEUS_SCHEMA"):
        PrometheusPublisher(CollectorRegistry())
