import threading
import uuid
from typing import Dict
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry, generate_latest
from src.service.data.datasources.data_source import DataSource
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.metric_value_carrier import MetricValueCarrier
from src.service.prometheus.prometheus_publisher import PrometheusPublisher
from src.service.prometheus.prometheus_scheduler import PrometheusScheduler


class MockMetricRequest(BaseMetricRequest):
    """Mock implementation of BaseMetricRequest for testing."""

    def __init__(
        self, model_id: str = "test_model", metric_name: str = "test_metric"
    ) -> None:
        super().__init__(
            model_id=model_id,
            metric_name=metric_name,
            request_name="test_request",
            batch_size=100,
        )

    def retrieve_tags(self) -> Dict[str, str]:
        return {"custom_tag": "custom_value"}


class TestPrometheusScheduler:
    @pytest.fixture
    def test_registry(self) -> CollectorRegistry:
        """Create a fresh Prometheus registry for each test."""
        return CollectorRegistry()

    @pytest.fixture
    def mock_publisher(self) -> Mock:
        return Mock(spec=PrometheusPublisher)

    @pytest.fixture
    def mock_data_source(self) -> Mock:
        """Create mock DataSource."""
        mock = Mock(spec=DataSource)
        mock.get_verified_models.return_value = ["model1", "model2"]
        mock.get_num_observations.return_value = 1000
        mock.has_recorded_inferences.return_value = True
        mock.get_organic_dataframe.return_value = pd.DataFrame(
            {"feature": [1, 2, 3], "target": [0, 1, 0]}
        )
        return mock

    @pytest.fixture
    def scheduler(
        self, mock_publisher: Mock, mock_data_source: Mock
    ) -> PrometheusScheduler:
        """Create PrometheusScheduler with mocked dependencies."""
        return PrometheusScheduler(
            publisher=mock_publisher, data_source=mock_data_source
        )

    @pytest.fixture
    def scheduler_with_publisher(
        self, test_registry: CollectorRegistry, mock_data_source: Mock
    ) -> PrometheusScheduler:
        """Create PrometheusScheduler with mocked dependencies."""
        return PrometheusScheduler(
            publisher=PrometheusPublisher(registry=test_registry),
            data_source=mock_data_source,
        )

    @pytest.fixture
    def mock_request(self) -> MockMetricRequest:
        """Create mock metric request."""
        return MockMetricRequest()

    def test_scheduler_initialization(self, scheduler: PrometheusScheduler) -> None:
        """Test scheduler initialization."""
        assert scheduler.requests == {}
        assert scheduler.has_logged_skipped_request_message == set()
        assert scheduler.metrics_directory is not None
        assert scheduler.publisher is not None
        assert scheduler.data_source is not None

    def test_service_config_parsing(self) -> None:
        """Test service configuration parsing."""
        with patch.dict(
            "os.environ",
            {"SERVICE_METRICS_SCHEDULE": "60s", "SERVICE_BATCH_SIZE": "200"},
        ):
            config = PrometheusScheduler.get_service_config()
            assert config["metrics_schedule"] == 60
            assert config["batch_size"] == 200

        with patch.dict("os.environ", {"SERVICE_METRICS_SCHEDULE": "5m"}):
            config = PrometheusScheduler.get_service_config()
            assert config["metrics_schedule"] == 300  # 300 seconds

        with patch.dict("os.environ", {"SERVICE_METRICS_SCHEDULE": "1h"}):
            config = PrometheusScheduler.get_service_config()
            assert config["metrics_schedule"] == 3600  # 3600 seconds

    def test_register_request(
        self, scheduler: PrometheusScheduler, mock_request: MockMetricRequest
    ) -> None:
        """Test registering a metric request."""
        test_id = uuid.uuid4()

        scheduler.register(metric_name="test_metric", id=test_id, request=mock_request)

        # Verify request is stored
        assert "test_metric" in scheduler.requests
        assert test_id in scheduler.requests["test_metric"]
        assert scheduler.requests["test_metric"][test_id] == mock_request

    def test_register_multiple_requests(self, scheduler: PrometheusScheduler) -> None:
        """Test registering multiple requests for same metric."""
        request1 = MockMetricRequest(model_id="model1")
        request2 = MockMetricRequest(model_id="model2")
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        scheduler.register(metric_name="test_metric", id=id1, request=request1)
        scheduler.register(metric_name="test_metric", id=id2, request=request2)

        assert len(scheduler.requests["test_metric"]) == 2
        assert scheduler.requests["test_metric"][id1] == request1
        assert scheduler.requests["test_metric"][id2] == request2

    def test_delete_request(
        self, scheduler: PrometheusScheduler, mock_request: MockMetricRequest
    ) -> None:
        """Test deleting a metric request."""
        test_id = uuid.uuid4()

        # Register then delete
        scheduler.register(metric_name="test_metric", id=test_id, request=mock_request)
        assert test_id in scheduler.requests["test_metric"]

        scheduler.delete(metric_name="test_metric", id=test_id)
        assert test_id not in scheduler.requests["test_metric"]

        # Verify publisher.remove_gauge was called
        scheduler.publisher.remove_gauge.assert_called_once_with("test_metric", test_id)

    def test_get_requests(
        self, scheduler: PrometheusScheduler, mock_request: MockMetricRequest
    ) -> None:
        """Test getting requests for a metric."""
        test_id = uuid.uuid4()
        scheduler.register(metric_name="test_metric", id=test_id, request=mock_request)

        requests = scheduler.get_requests("test_metric")
        assert len(requests) == 1
        assert test_id in requests
        assert requests[test_id] == mock_request

        # Test non-existent metric
        empty_requests = scheduler.get_requests("non_existent")
        assert len(empty_requests) == 0

    def test_get_all_requests_flat(self, scheduler: PrometheusScheduler) -> None:
        """Test getting all requests flattened."""
        request1 = MockMetricRequest(model_id="model1")
        request2 = MockMetricRequest(model_id="model2")
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        scheduler.register(metric_name="metric1", id=id1, request=request1)
        scheduler.register(metric_name="metric2", id=id2, request=request2)

        all_requests = scheduler.get_all_requests_flat()
        assert len(all_requests) == 2
        assert all_requests[id1] == request1
        assert all_requests[id2] == request2

    def test_has_requests(
        self, scheduler: PrometheusScheduler, mock_request: MockMetricRequest
    ) -> None:
        """Test has_requests functionality."""
        assert scheduler.has_requests() is False

        test_id = uuid.uuid4()
        scheduler.register(metric_name="test_metric", id=test_id, request=mock_request)
        assert scheduler.has_requests() is True

        scheduler.delete(metric_name="test_metric", id=test_id)
        assert scheduler.has_requests() is False

    def test_get_model_ids(self, scheduler: PrometheusScheduler) -> None:
        """Test getting unique model IDs."""
        request1 = MockMetricRequest(model_id="model1")
        request2 = MockMetricRequest(model_id="model2")
        request3 = MockMetricRequest(model_id="model1")  # Duplicate

        scheduler.register(metric_name="metric1", id=uuid.uuid4(), request=request1)
        scheduler.register(metric_name="metric2", id=uuid.uuid4(), request=request2)
        scheduler.register(metric_name="metric3", id=uuid.uuid4(), request=request3)

        model_ids = scheduler.get_model_ids()
        assert model_ids == {"model1", "model2"}

    def test_calculate_with_no_requests(self, scheduler: PrometheusScheduler) -> None:
        """Test calculate when no requests are registered."""
        scheduler.calculate()

        # Should still publish global metrics
        # MODEL_COUNT_TOTAL (1 publish) + MODEL_OBSERVATIONS_TOTAL per model (2 publishes)
        assert scheduler.publisher.gauge.call_count == 3

    def test_calculate_with_no_inferences(
        self, scheduler: PrometheusScheduler, mock_data_source: Mock
    ) -> None:
        """Test calculate when model has no recorded inferences."""
        mock_data_source.has_recorded_inferences.return_value = False

        mock_request = MockMetricRequest(model_id="model1")
        scheduler.register(
            metric_name="test_metric", id=uuid.uuid4(), request=mock_request
        )

        scheduler.calculate()

        # Should log skipped message
        assert "model1" in scheduler.has_logged_skipped_request_message

    def test_calculate_error_handling(
        self, scheduler: PrometheusScheduler, mock_data_source: Mock
    ) -> None:
        """Test error handling in calculate method."""
        # Make data source throw an exception
        mock_data_source.get_verified_models.side_effect = Exception("Test error")

        # Should not raise when throw_errors=False
        scheduler.calculate_manual(throw_errors=False)

        # Should raise when throw_errors=True
        with pytest.raises(Exception, match="Test error"):
            scheduler.calculate_manual(throw_errors=True)

    def test_thread_safety(self, scheduler: PrometheusScheduler) -> None:
        """Test thread safety of scheduler operations."""
        requests_created = {}
        threads = []

        def register_request(thread_id):
            request = MockMetricRequest(model_id=f"model_{thread_id}")
            test_id = uuid.uuid4()
            requests_created[thread_id] = (test_id, request)
            scheduler.register(
                metric_name=f"metric_{thread_id}", id=test_id, request=request
            )

        # Create multiple threads
        for i in range(10):
            thread = threading.Thread(target=register_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all requests were registered correctly
        for thread_id, (test_id, request) in requests_created.items():
            metric_name = f"metric_{thread_id}"
            assert metric_name in scheduler.requests
            assert test_id in scheduler.requests[metric_name]
            assert scheduler.requests[metric_name][test_id] == request

    def test_calculate_with_requests(
        self,
        scheduler_with_publisher: PrometheusScheduler,
        test_registry: CollectorRegistry,
    ) -> None:
        """Test calculate with registered requests."""
        # Mock the metrics directory calculator
        mock_calculator = Mock()
        mock_calculator.return_value = MetricValueCarrier(0.5)
        scheduler_with_publisher.metrics_directory.register(
            name="test_metric", calculator=mock_calculator
        )

        # Register a request
        mock_request = MockMetricRequest(model_id="model1", metric_name="test_metric")
        test_id = uuid.uuid4()
        scheduler_with_publisher.register(
            metric_name="test_metric", id=test_id, request=mock_request
        )

        scheduler_with_publisher.calculate()

        # Verify calculator was called
        mock_calculator.assert_called()

        # Verify metrics were published
        metrics_output = generate_latest(test_registry).decode("utf-8")
        assert "trustyai_test_metric" in metrics_output
        assert f'request="{test_id}"' in metrics_output
        assert "0.5" in metrics_output

    def test_calculate_with_multiple_requests_same_metric(
        self,
        scheduler_with_publisher: PrometheusScheduler,
        test_registry: CollectorRegistry,
    ) -> None:
        """Test calculate with multiple requests for the same metric type."""

        def mock_spd_calculator(
            df: pd.DataFrame, request: MockMetricRequest
        ) -> MetricValueCarrier:
            # Return different values based on model
            return MetricValueCarrier(0.1 if request.model_id == "model1" else 0.2)

        scheduler_with_publisher.metrics_directory.register(
            name="spd", calculator=mock_spd_calculator
        )

        # Register multiple requests
        request1 = MockMetricRequest(model_id="model1", metric_name="spd")
        request2 = MockMetricRequest(model_id="model2", metric_name="spd")
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        scheduler_with_publisher.register(metric_name="spd", id=id1, request=request1)
        scheduler_with_publisher.register(metric_name="spd", id=id2, request=request2)

        # Calculate metrics
        scheduler_with_publisher.calculate()

        # Get Prometheus output
        metrics_output = generate_latest(test_registry).decode("utf-8")

        # Verify both metrics exist
        assert "trustyai_spd" in metrics_output
        assert 'model="model1"' in metrics_output
        assert 'model="model2"' in metrics_output
        assert f'request="{id1}"' in metrics_output
        assert f'request="{id2}"' in metrics_output

        # Verify different values
        assert "0.1" in metrics_output
        assert "0.2" in metrics_output
