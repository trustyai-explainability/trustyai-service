"""Retain useful drift metadata in actual Prometheus samples."""

import importlib
import uuid

import pytest
from prometheus_client import CollectorRegistry

from trustyai_service.service.prometheus.gauge_config import GaugeConfig
from trustyai_service.service.prometheus.prometheus_publisher import PrometheusPublisher


@pytest.mark.parametrize(
    ("module", "request_class"),
    [
        ("compare_means", "CompareMeansMetricRequest"),
        ("jensen_shannon", "JensenShannonMetricRequest"),
        ("kolmogorov_smirnov", "KSTestMetricRequest"),
        ("kolmogorov_smirnov_streaming", "ApproxKSTestMetricRequest"),
        ("mmd", "MMDMetricRequest"),
    ],
)
def test_drift_metadata(module: str, request_class: str) -> None:
    """New batch labels coexist with reference, feature and accuracy labels."""
    metric_module = importlib.import_module(
        f"trustyai_service.endpoints.metrics.drift.{module}"
    )
    request = getattr(metric_module, request_class)(
        modelId="model", referenceTag="baseline", fitColumns=["feature"], batchSize=37
    )
    registry = CollectorRegistry()
    publisher = PrometheusPublisher(registry)
    publisher.gauge(
        GaugeConfig(
            model_name="model",
            request_id=uuid.uuid4(),
            request=request,
            named_values={"feature": 0.5},
        )
    )
    sample = next(sample for metric in registry.collect() for sample in metric.samples)
    assert sample.labels["batch_size"] == "37"
    assert sample.labels["referenceTag"] == "baseline"
    assert sample.labels["fitColumns"] == "feature"
    assert sample.labels["subcategory"] == "feature"
    if module == "kolmogorov_smirnov_streaming":
        assert sample.labels["epsilon"] == "0.01"
