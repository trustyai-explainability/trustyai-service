from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseMetricRequest(ABC):
    """
    Abstract base class for metric requests.
    """
    def __init__(self, model_id: str, metric_name: str, request_name: Optional[str], batch_size: Optional[int]) -> None:
        self.model_id: str = model_id
        self.metric_name: str = metric_name
        self.request_name: str = request_name
        self.batch_size: int = batch_size
    
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
