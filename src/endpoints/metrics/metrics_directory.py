import logging
from typing import Dict, Callable
from pandas import DataFrame

from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.metric_value_carrier import MetricValueCarrier

logger: logging.Logger = logging.getLogger(__name__)

class MetricsDirectory:
    def __init__(self) -> None:
        self.calculator_directory: Dict[str, Callable[[DataFrame, BaseMetricRequest], MetricValueCarrier]] = {}

    def register(self, name: str, calculator: Callable[[DataFrame, BaseMetricRequest], MetricValueCarrier]) -> None:
        if name not in self.calculator_directory:
            self.calculator_directory[name] = calculator
            logger.debug(f"Registered calculator for metric: {name}")
        

    def get_calculator(self, name: str) -> Callable[[DataFrame, BaseMetricRequest], MetricValueCarrier]:
        return self.calculator_directory.get(name)
