"""Base class for all metric request models."""

from abc import abstractmethod

from pydantic import BaseModel, ConfigDict


class BaseMetricRequest(BaseModel):
    """Abstract base class for metric requests."""

    # To allow extra fields to be set on instances
    model_config = ConfigDict(extra="allow")

    model_id: str
    metric_name: str
    request_name: str | None = None
    batch_size: int | None = None

    def retrieve_default_tags(self) -> dict[str, str]:
        """Retrieve default tags common to all metric requests."""
        output: dict[str, str] = {}
        if self.request_name is not None:
            output["requestName"] = self.request_name
        output["metricName"] = self.metric_name
        output["modelId"] = self.model_id
        return output

    @abstractmethod
    def retrieve_tags(self) -> dict[str, str]:
        """Retrieve metric-specific tags for this request."""
