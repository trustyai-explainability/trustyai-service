"""Prometheus label values for primitive and wrapped fairness requests."""

import uuid

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from trustyai_service.endpoints.metrics.fairness.group.utils import GroupMetricRequest
from trustyai_service.service.prometheus.gauge_config import GaugeConfig
from trustyai_service.service.prometheus.prometheus_publisher import PrometheusPublisher


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("male", '"male"'),
        ({"rawValueNode": {"value": "male"}}, '"male"'),
        (
            {"rawValueNodes": [{"value": "male"}, {"value": "other"}]},
            '["male","other"]',
        ),
        ({"reconciledType": [{"value": "male"}]}, '"male"'),
        (1, "1"),
    ],
)
def test_group_label_values(value: object, expected: str) -> None:
    """Equivalent request representations emit the same semantic label."""
    request = GroupMetricRequest.model_validate(
        {
            "modelId": "model",
            "metricName": "SPD",
            "protectedAttribute": "gender",
            "outcomeName": "approved",
            "privilegedAttribute": value,
            "unprivilegedAttribute": value,
            "favorableOutcome": value,
        }
    )
    tags = request.retrieve_tags()
    assert (
        tags["privileged"]
        == tags["unprivileged"]
        == tags["favorable_value"]
        == expected
    )
    registry = CollectorRegistry()
    publisher = PrometheusPublisher(registry)
    publisher.gauge(
        GaugeConfig(
            model_name="model", request_id=uuid.uuid4(), request=request, value=0.5
        )
    )
    sample = next(sample for metric in registry.collect() for sample in metric.samples)
    assert sample.labels["privileged"] == expected
    assert sample.labels["protectedAttribute"] == sample.labels["protected"] == "gender"
    assert "trustyai_spd" in generate_latest(registry).decode()
