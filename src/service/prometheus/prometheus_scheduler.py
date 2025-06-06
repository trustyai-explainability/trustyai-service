import logging
import os
import threading
import uuid
from collections import defaultdict
from typing import Dict, Optional, Set

from src.endpoints.metrics.metrics_directory import MetricsDirectory
from src.service.data.datasources.data_source import DataSource
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.payloads.metrics.request_reconciler import RequestReconciler
from src.service.prometheus.prometheus_publisher import PrometheusPublisher

logger: logging.Logger = logging.getLogger(__name__)


class PrometheusScheduler:

    @staticmethod
    def get_service_config() -> Dict:
        """Get service configuration from environment variables."""

        metrics_schedule = os.getenv("SERVICE_METRICS_SCHEDULE", "30s")
        batch_size = int(os.getenv("SERVICE_BATCH_SIZE", "100"))

        if metrics_schedule.endswith("s"):
            interval = int(metrics_schedule[:-1])
        elif metrics_schedule.endswith("m"):
            interval = int(metrics_schedule[:-1]) * 60
        elif metrics_schedule.endswith("h"):
            interval = int(metrics_schedule[:-1]) * 60 * 60
        else:
            interval = 30  # Default to 30 seconds

        return {"batch_size": batch_size, "metrics_schedule": interval}

    def __init__(
        self,
        publisher: Optional[PrometheusPublisher] = None,
        data_source: Optional[DataSource] = None,
    ) -> None:
        self.requests: Dict[str, Dict[uuid.UUID, BaseMetricRequest]] = defaultdict(dict)
        self.has_logged_skipped_request_message: Set[str] = set()
        self.metrics_directory: MetricsDirectory = MetricsDirectory()
        self.publisher: PrometheusPublisher = publisher or PrometheusPublisher()
        self.data_source: DataSource = data_source or DataSource()
        self.service_config: Dict = self.get_service_config()
        self._logged_skipped_request_message_lock: threading.Lock = threading.Lock()
        self._requests_lock: threading.Lock = threading.Lock()

    def get_requests(self, metric_name: str) -> Dict[uuid.UUID, BaseMetricRequest]:
        """Get all requests for a specific metric."""
        with self._requests_lock:
            return dict(self.requests.get(metric_name, {}))

    def get_all_requests_flat(self) -> Dict[uuid.UUID, BaseMetricRequest]:
        """Get all requests across all metrics as a flat dictionary."""
        result = {}
        with self._requests_lock:
            for metric_dict in self.requests.values():
                result.update(metric_dict)
        return result

    def calculate(self) -> None:
        """Calculate scheduled metrics."""
        self.calculate_manual(False)

    def calculate_manual(self, throw_errors: bool = True) -> None:
        """
        Calculate scheduled metrics.

        Args:
            throw_errors: If True, errors will be thrown. If False, they will just be logged.
        """
        try:
            verified_models = self.data_source.get_verified_models()

            # Global service statistic
            self.publisher.gauge(
                model_name="",
                id=PrometheusPublisher.generate_uuid("model_count"),
                metric_name="MODEL_COUNT_TOTAL",
                value=len(verified_models),
            )
            requested_models = self.get_model_ids()

            for model_id in verified_models:
                # Global model statistics
                total_observations = self.data_source.get_num_observations(model_id)
                self.publisher.gauge(
                    model_name=model_id,
                    id=PrometheusPublisher.generate_uuid(model_id),
                    metric_name="MODEL_OBSERVATIONS_TOTAL",
                    value=total_observations,
                )

                has_recorded_inferences = self.data_source.has_recorded_inferences(
                    model_id
                )

                if not has_recorded_inferences:
                    with self._logged_skipped_request_message_lock:
                        if model_id not in self.has_logged_skipped_request_message:
                            logger.info(
                                f"Skipping metric calculation for model={model_id}, "
                                "as no inference data has yet been recorded. "
                                "Once inference data arrives, metric calculation "
                                "will resume."
                            )
                            self.has_logged_skipped_request_message.add(model_id)
                        continue

                if self.has_requests() and model_id in requested_models:
                    requests_for_model = [
                        (req_id, request)
                        for req_id, request in self.get_all_requests_flat().items()
                        if request.model_id == model_id
                    ]

                    max_batch_size = max(
                        [request.batch_size for _, request in requests_for_model],
                        default=self.service_config.get("batch_size", 100),
                    )

                    df = self.data_source.get_organic_dataframe(
                        model_id, max_batch_size
                    )

                    for req_id, request in requests_for_model:
                        batch_size = min(request.batch_size, df.shape[0])
                        batch = df.tail(batch_size)

                        metric_name = request.metric_name
                        calculator = self.metrics_directory.get_calculator(metric_name)

                        if calculator:
                            value = calculator(batch, request)

                            if value.is_single():
                                self.publisher.gauge(
                                    model_name=model_id,
                                    id=req_id,
                                    request=request,
                                    value=value.get_value(),
                                )
                            else:
                                self.publisher.gauge(
                                    model_name=model_id,
                                    id=req_id,
                                    request=request,
                                    named_values=value.get_named_values(),
                                )
                        else:
                            logger.warning(
                                f"No calculator found for metric {metric_name}"
                            )

        except Exception as e:
            if throw_errors:
                raise e
            else:
                logger.error(f"Error calculating metrics: {e}")

    def register(
        self, metric_name: str, id: uuid.UUID, request: BaseMetricRequest
    ) -> None:
        """Register a metric request."""
        RequestReconciler.reconcile(request, self.data_source)
        with self._requests_lock:
            if metric_name not in self.requests:
                self.requests[metric_name] = {}
            self.requests[metric_name][id] = request

    def delete(self, metric_name: str, id: uuid.UUID) -> None:
        """Delete a metric request."""
        with self._requests_lock:
            if metric_name in self.requests and id in self.requests[metric_name]:
                del self.requests[metric_name][id]

        self.publisher.remove_gauge(metric_name, id)

    def has_requests(self) -> bool:
        """Check if there are any requests."""
        with self._requests_lock:
            return any(bool(requests) for requests in self.requests.values())

    def get_model_ids(self) -> Set[str]:
        """Get unique model IDs with registered Prometheus metrics."""
        model_ids = set()
        for request in self.get_all_requests_flat().values():
            model_ids.add(request.model_id)
        return model_ids
