import hashlib
import logging
import re
import threading
import uuid
from typing import Dict, List, Optional

from prometheus_client import CollectorRegistry, Gauge, REGISTRY
from src.service.constants import PROMETHEUS_METRIC_PREFIX
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.gauge_config import GaugeConfig

logger: logging.Logger = logging.getLogger(__name__)

# Prometheus metric name validation regex
# Must start with letter, underscore, or colon
# Can contain letters, numbers, underscores, and colons
# Lowercase only (shall we allow uppercase?)
PROMETHEUS_METRIC_NAME_REGEX = re.compile(r"^[a-z_:][a-z0-9_:]*$")


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
        # Track derived UUIDs created for named_values, keyed by root request ID.
        # Protected by _values_lock.
        self._derived_ids: Dict[uuid.UUID, List[uuid.UUID]] = {}

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

    def _create_or_update_gauge(self, name: str, tags: Dict[str, str], id: uuid.UUID) -> None:
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

                # IMPORTANT: Accessing private attributes of prometheus_client.Gauge
                # This is necessary because the prometheus_client library does not provide
                # public methods to:
                # 1. List existing metrics with their labels (_metrics.items())
                # 2. Access label names for a gauge (_labelnames)
                # We need this functionality to selectively remove gauge metrics
                # based on the "request" label matching the provided ID.
                # If prometheus_client adds public APIs for this in the future,
                # this should be refactored to use those instead.
                for labels, _ in gauge._metrics.items():
                    labels_dict = dict(zip(gauge._labelnames, labels))
                    if labels_dict.get("request") == str(id):
                        gauges_to_remove.append(labels)

                for to_remove in gauges_to_remove:
                    gauge.remove(*to_remove)

        # Clean up the root value and any derived values from named_values
        self._remove_value(id)
        with self._values_lock:
            for derived_id in self._derived_ids.pop(id, []):
                if derived_id in self.values:
                    del self.values[derived_id]

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

    def gauge(self, config: GaugeConfig) -> None:
        """Register a gauge metric from a validated GaugeConfig."""
        if config.request is not None:
            full_metric_name = self._get_full_metric_name(config.request.metric_name)

            if config.value is not None:
                self._set_value(config.request_id, config.value)
                tags = self._generate_tags(model_name=config.model_name, id=config.request_id, request=config.request)
                self._create_or_update_gauge(name=full_metric_name, tags=tags, id=config.request_id)
                logger.debug(
                    f"Scheduled request for {config.request.metric_name} id={config.request_id}, value={config.value}"
                )

            elif config.named_values is not None:
                derived_ids: List[uuid.UUID] = []
                for idx, (key, val) in enumerate(config.named_values.items()):
                    concat_string = f"{str(config.request_id)}{idx}"
                    new_id = self.generate_uuid(concat_string)
                    derived_ids.append(new_id)
                    self._set_value(new_id, val)

                    tags = self._generate_tags(
                        model_name=config.model_name, id=config.request_id, request=config.request
                    )
                    tags["subcategory"] = key
                    self._create_or_update_gauge(name=full_metric_name, tags=tags, id=new_id)

                # Track derived IDs so remove_gauge can clean them up
                with self._values_lock:
                    self._derived_ids[config.request_id] = derived_ids

                logger.debug(
                    f"Scheduled request for {config.request.metric_name} id={config.request_id}, value={config.named_values}"
                )

        else:
            # Simple gauge without request (metric_name required, validated by GaugeConfig)
            full_metric_name = self._get_full_metric_name(config.metric_name)
            self._set_value(config.request_id, config.value)

            tags = self._generate_tags(model_name=config.model_name, id=config.request_id)
            self._create_or_update_gauge(name=full_metric_name, tags=tags, id=config.request_id)

            logger.debug(f"Scheduled request for {config.metric_name} id={config.request_id}, value={config.value}")

    def _get_full_metric_name(self, metric_name: str) -> str:
        if not PROMETHEUS_METRIC_PREFIX.strip():
            raise ValueError("Prometheus metric prefix cannot be empty")
        if not metric_name.strip():
            raise ValueError("Metric name cannot be empty")

        full_name = f"{PROMETHEUS_METRIC_PREFIX}{metric_name.lower()}"
        # Validate the full metric name against the regex
        if not PROMETHEUS_METRIC_NAME_REGEX.match(full_name):
            raise ValueError(
                f"Invalid Prometheus metric name: '{metric_name}'. "
                f"Metric names must start with a lowercase letter, "
                f"underscore, or colon, and contain only lowercase letters, "
                f"numbers, underscores, and colons."
            )

        return full_name

    @staticmethod
    def generate_uuid(content: str) -> uuid.UUID:
        """
        Generates UUID with bytes to match Java's UUID.nameUUIDFromBytes()

        Uses MD5 for UUID generation only (not cryptographic security).
        This matches Java's implementation for cross-platform compatibility.
        """
        # lgtm[py/weak-sensitive-data-hashing]
        md5_hash = hashlib.md5(content.encode("utf-8"), usedforsecurity=False).digest()  # nosec B324 - MD5 used for UUID generation, not security
        return uuid.UUID(bytes=md5_hash, version=3)
