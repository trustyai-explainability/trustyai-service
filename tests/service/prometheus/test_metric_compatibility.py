"""Contract checks for uppercase metric identifiers and shared aliases."""

import importlib
import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, generate_latest

from trustyai_service.service.prometheus.gauge_config import GaugeConfig
from trustyai_service.service.prometheus.prometheus_publisher import PrometheusPublisher
from trustyai_service.service.prometheus.prometheus_scheduler import PrometheusScheduler


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
def test_alias_schedule_lifecycle(  # noqa: PLR0913 - route contract matrix
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    current: str,
    deprecated: str,
    fields: dict[str, str],
    *,
    reverse: bool,
) -> None:
    """Create, list, publish and delete through both spellings of a route."""
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
    assert request.metric_name.isupper()
    publisher.gauge(
        GaugeConfig(
            model_name="model", request_id=request_id, request=request, value=1.0
        )
    )
    scrape = generate_latest(registry).decode()
    assert f'request="{request_id}"' in scrape
    assert f'metricName="{request.metric_name}"' in scrape
    response = client.request(
        "DELETE",
        f"/metrics/{delete_route}/request",
        json={"requestId": str(request_id)},
    )
    assert response.status_code == 200
    assert not scheduler.get_all_requests_flat()
    assert f'request="{request_id}"' not in generate_latest(registry).decode()
