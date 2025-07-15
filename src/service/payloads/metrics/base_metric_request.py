from abc import abstractmethod
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict


class BaseMetricRequest(BaseModel):
    """
    Abstract base class for metric requests.
    """

    # To allow extra fields to be set on instances
    model_config = ConfigDict(extra="allow")

    model_id: str
    metric_name: str
    request_name: Optional[str] = None
    batch_size: Optional[int] = None

    def retrieve_default_tags(self) -> Dict[str, str]:
        output: Dict[str, str] = {}
        if self.request_name is not None:
            output["requestName"] = self.request_name
        output["metricName"] = self.metric_name
        output["modelId"] = self.model_id
        return output

    @abstractmethod
    def retrieve_tags(self) -> Dict[str, str]:
        pass
