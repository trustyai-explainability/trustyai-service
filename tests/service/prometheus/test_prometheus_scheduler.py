import asyncio
import uuid
from typing import Dict
from unittest.mock import AsyncMock, Mock, patch

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

    def __init__(self, model_id: str = "test_model", metric_name: str = "test_metric") -> None:
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
        mock.get_verified_models = AsyncMock(return_value=["model1", "model2"])
        mock.get_num_observations = AsyncMock(return_value=1000)
        mock.has_recorded_inferences = AsyncMock(return_value=True)
        mock.get_organic_dataframe = AsyncMock(return_value=pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]}))
        return mock

    @pytest.fixture
    def scheduler(self, mock_publisher: Mock, mock_data_source: Mock) -> PrometheusScheduler:
        """Create PrometheusScheduler with mocked dependencies."""
        return PrometheusScheduler(publisher=mock_publisher, data_source=mock_data_source)

    @pytest.fixture
    def scheduler_with_publisher(self, test_registry: CollectorRegistry, mock_data_source: Mock) -> PrometheusScheduler:
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

    def test_parse_interval_iso8601_seconds(self) -> None:
        """Test parsing ISO-8601 duration with seconds."""
        assert PrometheusScheduler._parse_schedule_interval("PT30S") == 30
        assert PrometheusScheduler._parse_schedule_interval("PT1S") == 1
        assert PrometheusScheduler._parse_schedule_interval("PT60S") == 60

    def test_parse_interval_iso8601_minutes(self) -> None:
        """Test parsing ISO-8601 duration with minutes."""
        assert PrometheusScheduler._parse_schedule_interval("PT5M") == 300
        assert PrometheusScheduler._parse_schedule_interval("PT1M") == 60
        assert PrometheusScheduler._parse_schedule_interval("PT30M") == 1800

    def test_parse_interval_iso8601_hours(self) -> None:
        """Test parsing ISO-8601 duration with hours."""
        assert PrometheusScheduler._parse_schedule_interval("PT2H") == 7200
        assert PrometheusScheduler._parse_schedule_interval("PT1H") == 3600
        assert PrometheusScheduler._parse_schedule_interval("PT24H") == 86400

    def test_parse_interval_iso8601_days(self) -> None:
        """Test parsing ISO-8601 duration with days."""
        assert PrometheusScheduler._parse_schedule_interval("P1D") == 86400
        assert PrometheusScheduler._parse_schedule_interval("P2D") == 172800

    def test_parse_interval_iso8601_combined(self) -> None:
        """Test parsing ISO-8601 duration with combined units."""
        # 1 day, 2 hours, 30 minutes, 45 seconds = 95445 seconds
        assert PrometheusScheduler._parse_schedule_interval("P1DT2H30M45S") == 95445
        # 1 hour and 30 minutes = 5400 seconds
        assert PrometheusScheduler._parse_schedule_interval("PT1H30M") == 5400

    def test_parse_interval_iso8601_fractional(self) -> None:
        """Test parsing ISO-8601 duration with fractional values."""
        assert PrometheusScheduler._parse_schedule_interval("PT0.5H") == 1800
        assert PrometheusScheduler._parse_schedule_interval("PT0.25M") == 15
        assert PrometheusScheduler._parse_schedule_interval("PT1.5S") == 1

    def test_parse_interval_simple_format_seconds(self) -> None:
        """Test parsing simple format with seconds (backward compatibility)."""
        assert PrometheusScheduler._parse_schedule_interval("30s") == 30
        assert PrometheusScheduler._parse_schedule_interval("1s") == 1
        assert PrometheusScheduler._parse_schedule_interval("60s") == 60

    def test_parse_interval_simple_format_minutes(self) -> None:
        """Test parsing simple format with minutes (backward compatibility)."""
        assert PrometheusScheduler._parse_schedule_interval("5m") == 300
        assert PrometheusScheduler._parse_schedule_interval("1m") == 60
        assert PrometheusScheduler._parse_schedule_interval("30m") == 1800

    def test_parse_interval_simple_format_hours(self) -> None:
        """Test parsing simple format with hours (backward compatibility)."""
        assert PrometheusScheduler._parse_schedule_interval("2h") == 7200
        assert PrometheusScheduler._parse_schedule_interval("1h") == 3600
        assert PrometheusScheduler._parse_schedule_interval("24h") == 86400

    def test_parse_interval_simple_format_days(self) -> None:
        """Test parsing simple format with days (backward compatibility)."""
        assert PrometheusScheduler._parse_schedule_interval("1d") == 86400
        assert PrometheusScheduler._parse_schedule_interval("2d") == 172800

    def test_parse_interval_invalid_format(self) -> None:
        """Test that invalid formats raise ValueError."""
        # "invalid" ends with 'd', so tries to parse as days and fails
        with pytest.raises(ValueError, match="Failed to parse schedule interval"):
            PrometheusScheduler._parse_schedule_interval("invalid")

        # "30x" doesn't match any suffix, goes to else clause
        with pytest.raises(ValueError, match="Invalid schedule format"):
            PrometheusScheduler._parse_schedule_interval("30x")

        # "abc" doesn't match any suffix, goes to else clause
        with pytest.raises(ValueError, match="Invalid schedule format"):
            PrometheusScheduler._parse_schedule_interval("abc")

    def test_parse_interval_empty_string(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            PrometheusScheduler._parse_schedule_interval("")

    def test_service_config_iso8601_format(self) -> None:
        """Test get_service_config with ISO-8601 format."""
        with patch.dict("os.environ", {"SERVICE_METRICS_SCHEDULE": "PT5M", "SERVICE_BATCH_SIZE": "200"}):
            config = PrometheusScheduler.get_service_config()
            assert config["metrics_schedule"] == 300  # 5 minutes
            assert config["batch_size"] == 200

        with patch.dict("os.environ", {"SERVICE_METRICS_SCHEDULE": "P1D"}):
            config = PrometheusScheduler.get_service_config()
            assert config["metrics_schedule"] == 86400  # 1 day

        with patch.dict("os.environ", {"SERVICE_METRICS_SCHEDULE": "PT1H30M"}):
            config = PrometheusScheduler.get_service_config()
            assert config["metrics_schedule"] == 5400  # 1.5 hours

    def test_service_config_invalid_schedule_falls_back(self) -> None:
        """Test that invalid schedule format falls back to default with warning."""
        with patch.dict("os.environ", {"SERVICE_METRICS_SCHEDULE": "invalid_format"}):
            config = PrometheusScheduler.get_service_config()
            assert config["metrics_schedule"] == 30  # Falls back to default

    @pytest.mark.asyncio
    async def test_register_request(self, scheduler: PrometheusScheduler, mock_request: MockMetricRequest) -> None:
        """Test registering a metric request."""
        test_id = uuid.uuid4()

        await scheduler.register(metric_name="test_metric", id=test_id, request=mock_request)

        # Verify request is stored
        assert "test_metric" in scheduler.requests
        assert test_id in scheduler.requests["test_metric"]
        assert scheduler.requests["test_metric"][test_id] == mock_request

    @pytest.mark.asyncio
    async def test_register_multiple_requests(self, scheduler: PrometheusScheduler) -> None:
        """Test registering multiple requests for same metric."""
        request1 = MockMetricRequest(model_id="model1")
        request2 = MockMetricRequest(model_id="model2")
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        await scheduler.register(metric_name="test_metric", id=id1, request=request1)
        await scheduler.register(metric_name="test_metric", id=id2, request=request2)

        assert len(scheduler.requests["test_metric"]) == 2
        assert scheduler.requests["test_metric"][id1] == request1
        assert scheduler.requests["test_metric"][id2] == request2

    @pytest.mark.asyncio
    async def test_delete_request(self, scheduler: PrometheusScheduler, mock_request: MockMetricRequest) -> None:
        """Test deleting a metric request."""
        test_id = uuid.uuid4()

        # Register then delete
        await scheduler.register(metric_name="test_metric", id=test_id, request=mock_request)
        assert test_id in scheduler.requests["test_metric"]

        await scheduler.delete(metric_name="test_metric", id=test_id)
        assert test_id not in scheduler.requests["test_metric"]

        # Verify publisher.remove_gauge was called
        scheduler.publisher.remove_gauge.assert_called_once_with("test_metric", test_id)

    @pytest.mark.asyncio
    async def test_get_requests(self, scheduler: PrometheusScheduler, mock_request: MockMetricRequest) -> None:
        """Test getting requests for a metric."""
        test_id = uuid.uuid4()
        await scheduler.register(metric_name="test_metric", id=test_id, request=mock_request)

        requests = scheduler.get_requests("test_metric")
        assert len(requests) == 1
        assert test_id in requests
        assert requests[test_id] == mock_request

        # Test non-existent metric
        empty_requests = scheduler.get_requests("non_existent")
        assert len(empty_requests) == 0

    @pytest.mark.asyncio
    async def test_get_all_requests_flat(self, scheduler: PrometheusScheduler) -> None:
        """Test getting all requests flattened."""
        request1 = MockMetricRequest(model_id="model1")
        request2 = MockMetricRequest(model_id="model2")
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        await scheduler.register(metric_name="metric1", id=id1, request=request1)
        await scheduler.register(metric_name="metric2", id=id2, request=request2)

        all_requests = scheduler.get_all_requests_flat()
        assert len(all_requests) == 2
        assert all_requests[id1] == request1
        assert all_requests[id2] == request2

    @pytest.mark.asyncio
    async def test_has_requests(self, scheduler: PrometheusScheduler, mock_request: MockMetricRequest) -> None:
        """Test has_requests functionality."""
        assert scheduler.has_requests() is False

        test_id = uuid.uuid4()
        await scheduler.register(metric_name="test_metric", id=test_id, request=mock_request)
        assert scheduler.has_requests() is True

        await scheduler.delete(metric_name="test_metric", id=test_id)
        assert scheduler.has_requests() is False

    @pytest.mark.asyncio
    async def test_get_model_ids(self, scheduler: PrometheusScheduler) -> None:
        """Test getting unique model IDs."""
        request1 = MockMetricRequest(model_id="model1")
        request2 = MockMetricRequest(model_id="model2")
        request3 = MockMetricRequest(model_id="model1")  # Duplicate

        await scheduler.register(metric_name="metric1", id=uuid.uuid4(), request=request1)
        await scheduler.register(metric_name="metric2", id=uuid.uuid4(), request=request2)
        await scheduler.register(metric_name="metric3", id=uuid.uuid4(), request=request3)

        model_ids = scheduler.get_model_ids()
        assert model_ids == {"model1", "model2"}

    @pytest.mark.asyncio
    async def test_calculate_with_no_requests(self, scheduler: PrometheusScheduler) -> None:
        """Test calculate when no requests are registered."""
        await scheduler.calculate()

        # Should still publish global metrics
        # MODEL_COUNT_TOTAL (1 publish) + MODEL_OBSERVATIONS_TOTAL per model (2 publishes)
        assert scheduler.publisher.gauge.call_count == 3

    @pytest.mark.asyncio
    async def test_calculate_with_no_inferences(self, scheduler: PrometheusScheduler, mock_data_source: Mock) -> None:
        """Test calculate when model has no recorded inferences."""
        mock_data_source.has_recorded_inferences.return_value = False

        mock_request = MockMetricRequest(model_id="model1")
        await scheduler.register(metric_name="test_metric", id=uuid.uuid4(), request=mock_request)

        await scheduler.calculate()

        # Should log skipped message
        assert "model1" in scheduler.has_logged_skipped_request_message

    @pytest.mark.asyncio
    async def test_calculate_error_handling(self, scheduler: PrometheusScheduler, mock_data_source: Mock) -> None:
        """Test error handling in calculate method."""
        # Make data source throw an exception
        mock_data_source.get_verified_models.side_effect = Exception("Test error")

        # Should not raise when throw_errors=False
        await scheduler.calculate_manual(throw_errors=False)

        # Should raise when throw_errors=True
        with pytest.raises(Exception, match="Test error"):
            await scheduler.calculate_manual(throw_errors=True)

    @pytest.mark.asyncio
    async def test_thread_safety(self, scheduler: PrometheusScheduler) -> None:
        """Test thread safety of scheduler operations."""
        requests_created = {}

        async def register_request(thread_id):
            request = MockMetricRequest(model_id=f"model_{thread_id}")
            test_id = uuid.uuid4()
            requests_created[thread_id] = (test_id, request)
            await scheduler.register(metric_name=f"metric_{thread_id}", id=test_id, request=request)

        await asyncio.gather(*[register_request(i) for i in range(10)])

        # Verify all requests were registered correctly
        for thread_id, (test_id, request) in requests_created.items():
            metric_name = f"metric_{thread_id}"
            assert metric_name in scheduler.requests
            assert test_id in scheduler.requests[metric_name]
            assert scheduler.requests[metric_name][test_id] == request

    @pytest.mark.asyncio
    async def test_calculate_with_requests(
        self,
        scheduler_with_publisher: PrometheusScheduler,
        test_registry: CollectorRegistry,
    ) -> None:
        """Test calculate with registered requests."""
        # Mock the metrics directory calculator
        mock_calculator = Mock()
        mock_calculator.return_value = MetricValueCarrier(0.5)
        scheduler_with_publisher.metrics_directory.register(name="test_metric", calculator=mock_calculator)

        # Register a request
        mock_request = MockMetricRequest(model_id="model1", metric_name="test_metric")
        test_id = uuid.uuid4()
        await scheduler_with_publisher.register(metric_name="test_metric", id=test_id, request=mock_request)

        await scheduler_with_publisher.calculate()

        # Verify calculator was called
        mock_calculator.assert_called()

        # Verify metrics were published
        metrics_output = generate_latest(test_registry).decode("utf-8")
        assert "trustyai_test_metric" in metrics_output
        assert f'request="{test_id}"' in metrics_output
        assert "0.5" in metrics_output

    @pytest.mark.asyncio
    async def test_calculate_with_multiple_requests_same_metric(
        self,
        scheduler_with_publisher: PrometheusScheduler,
        test_registry: CollectorRegistry,
    ) -> None:
        """Test calculate with multiple requests for the same metric type."""

        def mock_spd_calculator(df: pd.DataFrame, request: MockMetricRequest) -> MetricValueCarrier:
            # Return different values based on model
            return MetricValueCarrier(0.1 if request.model_id == "model1" else 0.2)

        scheduler_with_publisher.metrics_directory.register(name="spd", calculator=mock_spd_calculator)

        # Register multiple requests
        request1 = MockMetricRequest(model_id="model1", metric_name="spd")
        request2 = MockMetricRequest(model_id="model2", metric_name="spd")
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()

        await scheduler_with_publisher.register(metric_name="spd", id=id1, request=request1)
        await scheduler_with_publisher.register(metric_name="spd", id=id2, request=request2)

        # Calculate metrics
        await scheduler_with_publisher.calculate()

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
