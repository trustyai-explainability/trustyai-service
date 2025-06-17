import hashlib
import logging
import threading
import uuid
from typing import Dict, Optional

from prometheus_client import CollectorRegistry, Gauge, REGISTRY
from src.service.constants import PROMETHEUS_METRIC_PREFIX
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest

logger: logging.Logger = logging.getLogger(__name__)


class PrometheusPublisher:
    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.registry: CollectorRegistry = registry
        self.values: Dict[uuid.UUID, float] = {}
        self._values_lock: threading.RLock = threading.RLock()
        # Track gauges by metric name to avoid re-creating them
        # This is because prometheus_client doesn't expose
        # public methods to retrieve them with name.
        self._gauges: Dict[str, Gauge] = {}
        self._gauges_lock: threading.RLock = threading.RLock()

    def _get_value(self, id: uuid.UUID) -> float:
        with self._values_lock:
            return self.values[id]

    def _set_value(self, id: uuid.UUID, value: float) -> None:
        with self._values_lock:
            self.values[id] = value

    def _remove_value(self, id: uuid.UUID) -> None:
        with self._values_lock:
            if id in self.values:
                del self.values[id]

    def _create_or_update_gauge(
        self, name: str, tags: Dict[str, str], id: uuid.UUID
    ) -> None:
        with self._gauges_lock:
            # We need to track gauges because prometheus_client doesn't provide
            # a way to retrieve an existing gauge by name from the registry
            if name not in self._gauges:
                gauge = Gauge(
                    name=name,
                    documentation=f"TrustyAI metric: {name}",
                    labelnames=list(tags.keys()),
                    registry=self.registry,
                )
                self._gauges[name] = gauge

            gauge = self._gauges[name]

            gauge.labels(**tags).set(self._get_value(id))

    def remove_gauge(self, name: str, id: uuid.UUID) -> None:
        full_name = self._get_full_metric_name(name)

        with self._gauges_lock:
            gauges_to_remove = []

            if full_name in self._gauges:
                gauge = self._gauges[full_name]
                for labels, _ in gauge._metrics.items():
                    labels_dict = dict(zip(gauge._labelnames, labels))
                    if labels_dict.get("request") == str(id):
                        gauges_to_remove.append(labels)

                for to_remove in gauges_to_remove:
                    gauge.remove(*to_remove)

        self._remove_value(id)

    def _generate_tags(
        self,
        model_name: str,
        id: uuid.UUID,
        request: Optional[BaseMetricRequest] = None,
    ) -> Dict[str, str]:
        tags: Dict[str, str] = {}

        if request is not None:
            tags.update(request.retrieve_default_tags())
            tags.update(request.retrieve_tags())
        elif model_name:
            tags["model"] = model_name

        tags["request"] = str(id)
        return tags

    def gauge(
        self,
        model_name: str,
        id: uuid.UUID,
        request: Optional[BaseMetricRequest] = None,
        value: Optional[float] = None,
        named_values: Optional[Dict[str, float]] = None,
        metric_name: Optional[str] = None,
    ) -> None:
        """
        Register a gauge metric with multiple possible parameter combinations:
        - gauge(model_name, id, request, value)
        - gauge(model_name, id, request, named_values)
        - gauge(model_name, id, value, metric_name)
        """
        if request is not None:
            full_metric_name = self._get_full_metric_name(request.metric_name)

            if value is not None:
                # gauge(model_name, id, request, value)
                self._set_value(id, value)
                tags = self._generate_tags(
                    model_name=model_name, id=id, request=request
                )
                self._create_or_update_gauge(name=full_metric_name, tags=tags, id=id)
                logger.debug(
                    f"Scheduled request for {request.metric_name} id={id}, value={value}"
                )

            elif named_values is not None:
                # gauge(model_name, id, request, named_values)
                for idx, (key, val) in enumerate(named_values.items()):
                    concat_string = f"{str(id)}{idx}"
                    new_id = self.generate_uuid(concat_string)
                    self._set_value(new_id, val)

                    tags = self._generate_tags(
                        model_name=model_name, id=id, request=request
                    )
                    tags["subcategory"] = key
                    self._create_or_update_gauge(
                        name=full_metric_name, tags=tags, id=new_id
                    )
                logger.debug(
                    f"Scheduled request for {request.metric_name} id={id}, value={named_values}"
                )
            else:
                raise ValueError("Either 'value' or 'named_values' must be provided")

        elif metric_name is not None and value is not None:
            # gauge(model_name, id, value, metric_name)
            full_metric_name = self._get_full_metric_name(metric_name)
            self._set_value(id, value)

            tags = self._generate_tags(model_name=model_name, id=id)
            self._create_or_update_gauge(name=full_metric_name, tags=tags, id=id)

            logger.debug(f"Scheduled request for {metric_name} id={id}, value={value}")

        else:
            raise ValueError(
                "Either 'request' or 'metric_name' & 'value' must be provided"
            )

    def _get_full_metric_name(self, metric_name: str) -> str:
        return f"{PROMETHEUS_METRIC_PREFIX}{metric_name.lower()}"

    @staticmethod
    def generate_uuid(content: str) -> uuid.UUID:
        """
        Generates UUID with bytes to match Java's UUID.nameUUIDFromBytes()
        """
        md5_hash = hashlib.md5(content.encode("utf-8")).digest()
        return uuid.UUID(bytes=md5_hash, version=3)
