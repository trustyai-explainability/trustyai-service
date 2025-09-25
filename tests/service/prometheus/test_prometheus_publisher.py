import threading
import uuid
from typing import Dict

import pytest

from prometheus_client import CollectorRegistry, Gauge
from src.service.constants import PROMETHEUS_METRIC_PREFIX
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest

from src.service.prometheus.prometheus_publisher import PrometheusPublisher


class MockMetricRequest(BaseMetricRequest):
    """Mock implementation of BaseMetricRequest for testing."""

    def __init__(self, model_id: str = "test_model", metric_name: str = "test_metric") -> None:
        super().__init__(
            model_id=model_id,
            metric_name=metric_name,
            request_name="test_request",
            batch_size=100,
        )

    def retrieve_tags(self) -> Dict[str, str]:
        return {"custom_tag": "custom_value"}


class TestPrometheusPublisher:
    @pytest.fixture
    def test_registry(self) -> CollectorRegistry:
        """Create a fresh registry for each test."""
        return CollectorRegistry()

    @pytest.fixture
    def publisher(self, test_registry: CollectorRegistry) -> PrometheusPublisher:
        """Create PrometheusPublisher with test registry."""
        return PrometheusPublisher(registry=test_registry)

    @pytest.fixture
    def mock_request(self) -> MockMetricRequest:
        """Create a mock metric request."""
        return MockMetricRequest()

    def test_single_value_gauge_with_request(self, publisher: PrometheusPublisher, mock_request: MockMetricRequest):
        """Test publishing single value gauge with request object."""
        test_id = uuid.uuid4()
        test_value = 0.5

        publisher.gauge(model_name="test_model", id=test_id, value=test_value, request=mock_request)

        # Verify value is stored
        assert test_id in publisher.values
        assert publisher.values[test_id] == test_value

        # Verify gauge is created in registry
        metric_name = f"{PROMETHEUS_METRIC_PREFIX}{mock_request.metric_name}"
        assert metric_name in publisher.registry._names_to_collectors

        # Verify gauge has correct labels
        expected_labels: list[str] = list(
            mock_request.retrieve_default_tags().keys()
        )  # default labels (metricName, modelId, requestName)
        expected_labels.extend(list(mock_request.retrieve_tags().keys()))  # custom tags (custom_tag)
        expected_labels.append("request")  # request id (request)

        gauge: Gauge = publisher.registry._names_to_collectors[metric_name]
        assert set(gauge._labelnames) == set(expected_labels)

    def test_named_values_gauge_with_request(self, publisher: PrometheusPublisher, mock_request: MockMetricRequest):
        """Test publishing named values gauge with request object."""
        test_id = uuid.uuid4()
        test_named_values = {"feature1": 0.3, "feature2": 0.7}

        publisher.gauge(
            model_name="test_model",
            id=test_id,
            named_values=test_named_values,
            request=mock_request,
        )

        # Verify multiple values are stored (one for each named value)
        assert test_named_values["feature1"] in publisher.values.values()
        assert test_named_values["feature2"] in publisher.values.values()

        # Verify gauge is created with subcategory label
        metric_name = f"{PROMETHEUS_METRIC_PREFIX}{mock_request.metric_name}"
        gauge: Gauge = publisher.registry._names_to_collectors[metric_name]

        assert "subcategory" in gauge._labelnames

        subcategory_value_map = {}
        for metric in gauge.collect():
            for sample in metric.samples:
                subcategory_value_map[sample.labels["subcategory"]] = sample.value
        assert subcategory_value_map == test_named_values

    def test_simple_gauge_without_request(self, publisher: PrometheusPublisher):
        """Test publishing simple gauge without request object."""
        test_id = uuid.uuid4()
        test_value = 1.0

        publisher.gauge(
            model_name="test_model",
            id=test_id,
            value=test_value,
            metric_name="simple_metric",
        )

        # Verify value is stored
        assert test_id in publisher.values
        assert publisher.values[test_id] == test_value

        # Verify gauge is created
        metric_name = f"{PROMETHEUS_METRIC_PREFIX}simple_metric"
        assert metric_name in publisher.registry._names_to_collectors

    def test_thread_safety(self, publisher: PrometheusPublisher, mock_request: MockMetricRequest):
        """Test thread safety of gauge operations."""
        test_values = {}
        threads: list[threading.Thread] = []

        def create_gauge(thread_id):
            test_id = uuid.uuid4()
            test_value = thread_id * 0.1
            test_values[thread_id] = (test_id, test_value)
            publisher.gauge(
                model_name=f"model_{thread_id}",
                id=test_id,
                value=test_value,
                request=mock_request,
            )

        # Create multiple threads
        for i in range(10):
            thread = threading.Thread(target=create_gauge, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all values were stored correctly
        for thread_id, (test_id, test_value) in test_values.items():
            assert test_id in publisher.values
            assert publisher.values[test_id] == test_value

    def test_remove_gauge(self, publisher: PrometheusPublisher, mock_request: MockMetricRequest):
        """Test gauge removal functionality."""
        test_id = uuid.uuid4()
        test_value = 0.5

        # Create gauge
        publisher.gauge(model_name="test_model", id=test_id, value=test_value, request=mock_request)
        assert test_id in publisher.values

        # Remove gauge
        publisher.remove_gauge(name=mock_request.metric_name, id=test_id)
        assert test_id not in publisher.values

    def test_duplicate_gauge_creation(self, publisher: PrometheusPublisher, mock_request: MockMetricRequest):
        """Test that creating gauges with same name but different labels."""
        test_id1 = uuid.uuid4()
        test_id2 = uuid.uuid4()

        # Create first gauge
        publisher.gauge(model_name="test_model", id=test_id1, value=0.5, request=mock_request)

        # Create second gauge with same metric name - should work with same labels
        publisher.gauge(model_name="test_model", id=test_id2, value=0.7, request=mock_request)

        # Both values should be stored
        assert test_id1 in publisher.values
        assert test_id2 in publisher.values
        assert publisher.values[test_id1] == 0.5
        assert publisher.values[test_id2] == 0.7

    def test_tag_generation_with_request(self, publisher: PrometheusPublisher, mock_request: MockMetricRequest):
        """Test tag generation with request object."""
        test_id = uuid.uuid4()
        tags = publisher._generate_tags(model_name="test_model", id=test_id, request=mock_request)

        expected_tags = {
            "metricName": mock_request.metric_name,
            "modelId": "test_model",
            "requestName": mock_request.request_name,
            "custom_tag": "custom_value",
            "request": str(test_id),
        }

        assert tags == expected_tags

    def test_tag_generation_without_request(self, publisher: PrometheusPublisher):
        """Test tag generation without request object."""
        test_id = uuid.uuid4()
        tags = publisher._generate_tags(model_name="test_model", id=test_id)

        expected_tags = {"model": "test_model", "request": str(test_id)}

        assert tags == expected_tags

    def test_invalid_gauge_parameters(self, publisher: PrometheusPublisher, mock_request: MockMetricRequest):
        """Test error handling for invalid parameters."""
        test_id = uuid.uuid4()

        # Test missing both value and named_values
        with pytest.raises(ValueError, match="Either 'value' or 'named_values' must be provided"):
            publisher.gauge(model_name="test_model", id=test_id, request=mock_request)

        # Test missing both request and metric_name
        with pytest.raises(
            ValueError,
            match="Either 'request' or 'metric_name' & 'value' must be provided",
        ):
            publisher.gauge(model_name="test_model", id=test_id, value=0.5)

    def test_valid_metric_names(self, publisher: PrometheusPublisher):
        """Test validation of valid Prometheus metric names."""
        valid_names = [
            "simple_metric",
            "metric_with_numbers123",
            "UPPERCASE_METRIC",  # we convert to lowercase
            "metric:with:colons",
            "_underscore_start",
            "a1b2c3",
            "metric_name_with_3underscores",
        ]

        for name in valid_names:
            full_name = publisher._get_full_metric_name(name)
            expected = f"{PROMETHEUS_METRIC_PREFIX}{name.lower()}"
            assert full_name == expected

    def test_invalid_metric_names(self, publisher: PrometheusPublisher):
        """Test validation of invalid Prometheus metric names."""
        invalid_names = [
            "metric-with-hyphens",
            "metric with spaces",
            "metric@with@symbols",
            "metric.with.dots",
            "metric#with#hash",
            "metric%with%percent",
            "-starts_with_hyphen",
            "metric/with/slashes",
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid Prometheus metric name"):
                publisher._get_full_metric_name(name)

    def test_empty_metric_name_validation(self, publisher: PrometheusPublisher):
        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            publisher._get_full_metric_name("")

    def test_gauge_creation_with_invalid_metric_name(
        self, publisher: PrometheusPublisher, mock_request: MockMetricRequest
    ):
        mock_request.metric_name = "invalid-metric-name"

        test_id = uuid.uuid4()
        test_value = 0.5

        with pytest.raises(ValueError, match="Invalid Prometheus metric name"):
            publisher.gauge(
                model_name="test_model",
                id=test_id,
                value=test_value,
                request=mock_request,
            )
