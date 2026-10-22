"""Prometheus label values for primitive and wrapped fairness requests."""

import uuid

import pytest
from prometheus_client import CollectorRegistry, generate_latest
from pydantic import ValidationError

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


def _spd_request(**overrides: object) -> dict[str, object]:
    """Build a minimal SPD request payload, overriding selected fields."""
    payload: dict[str, object] = {
        "modelId": "model",
        "metricName": "SPD",
        "protectedAttribute": "gender",
        "outcomeName": "approved",
        "privilegedAttribute": "male",
        "unprivilegedAttribute": "female",
        "favorableOutcome": 1,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "field",
    ["privilegedAttribute", "unprivilegedAttribute", "favorableOutcome"],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_group_rejects_non_finite_values(field: str, bad: float) -> None:
    """NaN and infinity are rejected at validation time, not during serialization."""
    with pytest.raises(ValidationError):
        GroupMetricRequest.model_validate(_spd_request(**{field: bad}))


def test_group_batch_size_label_never_renders_none() -> None:
    """An explicit null batch size resolves to the effective default in the label."""
    request = GroupMetricRequest.model_validate(_spd_request(batchSize=None))
    assert request.retrieve_tags()["batch_size"] == "100"
