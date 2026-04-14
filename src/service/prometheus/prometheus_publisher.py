"""Prometheus metric publisher for exposing calculated metrics."""

import hashlib
import logging
import re
import threading
import uuid

from prometheus_client import REGISTRY, CollectorRegistry, Gauge

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
    """Publishes metric values to Prometheus registry."""

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        """Initialize Prometheus publisher.

        :param registry: Prometheus collector registry to use
        """
        self.registry: CollectorRegistry = registry
        self.values: dict[uuid.UUID, float] = {}
        self._values_lock: threading.RLock = threading.RLock()
        # Track gauges by metric name to avoid re-creating them
        # This is because prometheus_client doesn't expose
        # public methods to retrieve them with name.
        self._gauges: dict[str, Gauge] = {}
        self._gauges_lock: threading.RLock = threading.RLock()

    def _get_value(self, request_id: uuid.UUID) -> float:
        with self._values_lock:
            return self.values[request_id]

    def _set_value(self, request_id: uuid.UUID, value: float) -> None:
        with self._values_lock:
            self.values[request_id] = value

    def _remove_value(self, request_id: uuid.UUID) -> None:
        with self._values_lock:
            if request_id in self.values:
                del self.values[request_id]

    def _create_or_update_gauge(
        self, name: str, tags: dict[str, str], request_id: uuid.UUID
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

            gauge.labels(**tags).set(self._get_value(request_id))

    def remove_gauge(self, name: str, request_id: uuid.UUID) -> None:
        """Remove a gauge metric from the registry.

        :param name: Metric name
        :param request_id: Request identifier for the metric
        """
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
                for labels in gauge._metrics:
                    labels_dict = dict(zip(gauge._labelnames, labels, strict=False))
                    if labels_dict.get("request") == str(request_id):
                        gauges_to_remove.append(labels)

                for to_remove in gauges_to_remove:
                    gauge.remove(*to_remove)

        self._remove_value(request_id)

    def _generate_tags(
        self,
        model_name: str,
        request_id: uuid.UUID,
        request: BaseMetricRequest | None = None,
    ) -> dict[str, str]:
        tags: dict[str, str] = {}

        if request is not None:
            tags.update(request.retrieve_default_tags())
            tags.update(request.retrieve_tags())
        elif model_name:
            tags["model"] = model_name

        tags["request"] = str(request_id)
        return tags

    def gauge(self, config: GaugeConfig) -> None:
        """Register a gauge metric.

        Supports three usage patterns via GaugeConfig:
        - gauge(GaugeConfig(model_name, request_id, request, value))
        - gauge(GaugeConfig(model_name, request_id, request, named_values))
        - gauge(GaugeConfig(model_name, request_id, value, metric_name))

        Args:
            config: Configuration object containing all parameters for gauge publishing

        """
        if config.request is not None:
            full_metric_name = self._get_full_metric_name(config.request.metric_name)

            if config.value is not None:
                self._set_value(config.request_id, config.value)
                tags = self._generate_tags(
                    model_name=config.model_name,
                    request_id=config.request_id,
                    request=config.request,
                )
                self._create_or_update_gauge(
                    name=full_metric_name, tags=tags, request_id=config.request_id
                )
                logger.debug(
                    "Scheduled request for %s id=%s, value=%s",
                    config.request.metric_name,
                    config.request_id,
                    config.value,
                )

            elif config.named_values is not None:
                for idx, (key, val) in enumerate(config.named_values.items()):
                    concat_string = f"{config.request_id!s}{idx}"
                    new_id = self.generate_uuid(concat_string)
                    self._set_value(new_id, val)

                    tags = self._generate_tags(
                        model_name=config.model_name,
                        request_id=config.request_id,
                        request=config.request,
                    )
                    tags["subcategory"] = key
                    self._create_or_update_gauge(
                        name=full_metric_name, tags=tags, request_id=new_id
                    )
                logger.debug(
                    "Scheduled request for %s id=%s, value=%s",
                    config.request.metric_name,
                    config.request_id,
                    config.named_values,
                )
            else:
                # GaugeConfig validation should prevent this
                msg = "GaugeConfig validation failed: either 'value' or 'named_values' must be provided"
                raise AssertionError(msg)

        elif config.metric_name is not None and config.value is not None:
            full_metric_name = self._get_full_metric_name(config.metric_name)
            self._set_value(config.request_id, config.value)

            tags = self._generate_tags(
                model_name=config.model_name, request_id=config.request_id
            )
            self._create_or_update_gauge(
                name=full_metric_name, tags=tags, request_id=config.request_id
            )

            logger.debug(
                "Scheduled request for %s id=%s, value=%s",
                config.metric_name,
                config.request_id,
                config.value,
            )

        else:
            # GaugeConfig validation should prevent this
            msg = "GaugeConfig validation failed: either 'request' or 'metric_name' & 'value' must be provided"
            raise AssertionError(msg)

    def _get_full_metric_name(self, metric_name: str) -> str:
        if not PROMETHEUS_METRIC_PREFIX.strip():
            msg = "Prometheus metric prefix cannot be empty"
            raise ValueError(msg)
        if not metric_name.strip():
            msg = "Metric name cannot be empty"
            raise ValueError(msg)

        full_name = f"{PROMETHEUS_METRIC_PREFIX}{metric_name.lower()}"
        # Validate the full metric name against the regex
        if not PROMETHEUS_METRIC_NAME_REGEX.match(full_name):
            msg = (
                f"Invalid Prometheus metric name: '{metric_name}'. "
                f"Metric names must start with a lowercase letter, "
                f"underscore, or colon, and contain only lowercase letters, "
                f"numbers, underscores, and colons."
            )
            raise ValueError(msg)

        return full_name

    @staticmethod
    def generate_uuid(content: str) -> uuid.UUID:
        """Generate UUID with bytes to match Java's UUID.nameUUIDFromBytes().

        Uses MD5 for UUID generation only (not cryptographic security).
        This matches Java's implementation for cross-platform
        compatibility.
        """
        md5_hash = hashlib.md5(content.encode("utf-8"), usedforsecurity=False).digest()  # nosec B324 - MD5 used for UUID generation, not security
        return uuid.UUID(bytes=md5_hash, version=3)
