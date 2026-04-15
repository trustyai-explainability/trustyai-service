"""Configuration objects for Prometheus gauge publishing."""

import uuid

from pydantic import BaseModel, model_validator

from src.service.payloads.metrics.base_metric_request import BaseMetricRequest


class GaugeConfig(BaseModel):
    """Configuration for publishing a Prometheus gauge metric.

    Supports three usage patterns:
    1. Single value with request object
    2. Named values (dict) with request object
    3. Simple value without request (requires metric_name)
    """

    model_name: str
    request_id: uuid.UUID
    request: BaseMetricRequest | None = None
    value: float | None = None
    named_values: dict[str, float] | None = None
    metric_name: str | None = None

    @model_validator(mode="after")
    def validate_gauge_parameters(self) -> "GaugeConfig":
        """Validate that the gauge configuration is valid.

        Rules:
        1. If request is provided: must have either value or named_values
        2. If request is None: must have both metric_name and value
        """
        if self.request is not None:
            # Request-based gauge: must have value or named_values
            if self.value is None and self.named_values is None:
                msg = "Either 'value' or 'named_values' must be provided"
                raise ValueError(msg)
        else:
            # Simple gauge: must have both metric_name and value
            if self.metric_name is None:
                msg = "Either 'request' or 'metric_name' must be provided"
                raise ValueError(msg)
            if self.value is None:
                msg = "Value must be provided when using metric_name"
                raise ValueError(msg)

        return self
