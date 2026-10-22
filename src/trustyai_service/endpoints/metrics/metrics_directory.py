"""Registry for metric calculation functions and metadata."""

import logging
from collections.abc import Callable

from pandas import DataFrame

from trustyai_service.service.payloads.metrics.base_metric_request import (
    BaseMetricRequest,
)
from trustyai_service.service.prometheus.metric_value_carrier import MetricValueCarrier

logger: logging.Logger = logging.getLogger(__name__)


class MetricsDirectory:
    """Registry for metric calculation functions and metadata."""

    def __init__(self) -> None:
        """Initialize the metrics directory with an empty calculator registry."""
        self.calculator_directory: dict[
            str, Callable[[DataFrame, BaseMetricRequest], MetricValueCarrier]
        ] = {}

    def register(
        self,
        name: str,
        calculator: Callable[[DataFrame, BaseMetricRequest], MetricValueCarrier],
    ) -> None:
        """Register a metric calculator function.

        :param name: Metric name identifier
        :param calculator: Function that computes the metric
        """
        if name not in self.calculator_directory:
            self.calculator_directory[name] = calculator
            logger.debug("Registered calculator for metric: %s", name)
        else:
            logger.warning(
                "Attempted to register duplicate calculator for metric: %s. Ignoring duplicate registration.",
                name,
            )

    def get_calculator(
        self, name: str
    ) -> Callable[[DataFrame, BaseMetricRequest], MetricValueCarrier]:
        """Get the calculator function for a metric.

        :param name: Metric name identifier
        :return: Calculator function or None if not found
        """
        return self.calculator_directory.get(name)
