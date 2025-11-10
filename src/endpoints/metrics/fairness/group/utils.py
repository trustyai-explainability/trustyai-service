from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
import logging
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.shared_prometheus_scheduler import get_shared_prometheus_scheduler
from src.service.data.shared_data_source import get_shared_data_source
from src.service.prometheus.metric_value_carrier import MetricValueCarrier

logger = logging.getLogger(__name__)

class ReconcilableFeature(BaseModel):
    rawValueNodes: Optional[List[Dict[str, Any]]] = None
    rawValueNode: Optional[Dict[str, Any]] = None
    reconciledType: Optional[List[Dict[str, Any]]] = None
    multipleValued: Optional[bool] = None


class ReconcilableOutput(BaseModel):
    rawValueNodes: Optional[List[Dict[str, Any]]] = None
    rawValueNode: Optional[Dict[str, Any]] = None
    reconciledType: Optional[List[Dict[str, Any]]] = None
    multipleValued: Optional[bool] = None


class GroupMetricRequest(BaseMetricRequest):
    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_id: str = Field(alias="modelId")
    metric_name: Optional[str] = Field(default=None, alias="metricName")  # Will be set by endpoint
    request_name: Optional[str] = Field(default=None, alias="requestName")
    batch_size: Optional[int] = Field(default=100, alias="batchSize")

    # SPD-specific fields
    protected_attribute: str = Field(alias="protectedAttribute")
    outcome_name: str = Field(alias="outcomeName")
    privileged_attribute: Union[ReconcilableFeature, int, float, str] = Field(alias="privilegedAttribute")
    unprivileged_attribute: Union[ReconcilableFeature, int, float, str] = Field(alias="unprivilegedAttribute")
    favorable_outcome: Union[ReconcilableOutput, int, float, str] = Field(alias="favorableOutcome")
    threshold_delta: Optional[float] = Field(default=None, alias="thresholdDelta")

    def retrieve_tags(self) -> Dict[str, str]:
        """Retrieve tags for this SPD metric request."""
        tags = self.retrieve_default_tags()
        tags["protectedAttribute"] = self.protected_attribute
        tags["outcomeName"] = self.outcome_name
        return tags



class GroupDefinitionRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    protectedAttribute: str
    outcomeName: str
    privilegedAttribute: Union[ReconcilableFeature, int, float, str]
    unprivilegedAttribute: Union[ReconcilableFeature, int, float, str]
    favorableOutcome: Union[ReconcilableOutput, int, float, str]
    thresholdDelta: Optional[float] = None
    metricValue: Dict[str, Any]

    # Additional snake_case properties for metrics processing
    @property
    def protected_attribute(self) -> str:
        return self.protectedAttribute

    @property
    def outcome_name(self) -> str:
        return self.outcomeName

    @property
    def privileged_attribute(self) -> Union[ReconcilableFeature, int, float, str]:
        return self.privilegedAttribute

    @property
    def unprivileged_attribute(self) -> Union[ReconcilableFeature, int, float, str]:
        return self.unprivilegedAttribute

    @property
    def favorable_outcome(self) -> Union[ReconcilableOutput, int, float, str]:
        return self.favorableOutcome

    @property
    def threshold_delta(self) -> Optional[float]:
        return self.thresholdDelta


class ScheduleId(BaseModel):
    requestId: str


def get_prometheus_scheduler():
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()

def get_data_source():
    """Get the shared data source instance."""
    return get_shared_data_source()


def calculate_fairness_metric(dataframe: pd.DataFrame, request: GroupMetricRequest) -> MetricValueCarrier:
    """
    Calculate a fairness metric for the given dataframe and request.
    This function is used to calculate the SPD and DIR metrics.
    Args:
        dataframe: The dataframe to calculate the metric on.
        request: The Group Fairness request object.
    Returns:
        The MetricValueCarrier containing the fairness metric value.
    """
    # Extract data from the reconciled request
    # model_id = request.modelId if hasattr(request, 'modelId') else request.model_id
    protected_attr = request.protectedAttribute if hasattr(request, 'protectedAttribute') else request.protected_attribute
    outcome_name = request.outcomeName if hasattr(request, 'outcomeName') else request.outcome_name
    metric_name = request.metric_name if hasattr(request, 'metricName') else request.metric_name

    # Handle different types of privilege/unprivilege attributes
    if hasattr(request, 'privilegedAttribute') and hasattr(request.privilegedAttribute, 'reconciledType'):
        # Complex reconciled type
        privileged_values = [item.get('value') for item in request.privilegedAttribute.reconciledType if 'value' in item]
    else:
        # Simple value
        privileged_attr = getattr(request, 'privilegedAttribute', getattr(request, 'privileged_attribute', None))
        privileged_values = [privileged_attr] if not isinstance(privileged_attr, list) else privileged_attr

    if hasattr(request, 'unprivilegedAttribute') and hasattr(request.unprivilegedAttribute, 'reconciledType'):
        unprivileged_values = [item.get('value') for item in request.unprivilegedAttribute.reconciledType if 'value' in item]
    else:
        unprivileged_attr = getattr(request, 'unprivilegedAttribute', getattr(request, 'unprivileged_attribute', None))
        unprivileged_values = [unprivileged_attr] if not isinstance(unprivileged_attr, list) else unprivileged_attr

    if hasattr(request, 'favorableOutcome') and hasattr(request.favorableOutcome, 'reconciledType'):
        favorable_values = [item.get('value') for item in request.favorableOutcome.reconciledType if 'value' in item]
    else:
        favorable_attr = getattr(request, 'favorableOutcome', getattr(request, 'favorable_outcome', None))
        favorable_values = [favorable_attr] if not isinstance(favorable_attr, list) else favorable_attr

    # Filter the dataframe into privileged and unprivileged groups
    privileged_mask = dataframe[protected_attr].isin(privileged_values)
    unprivileged_mask = dataframe[protected_attr].isin(unprivileged_values)

    privileged_data = dataframe[privileged_mask]
    unprivileged_data = dataframe[unprivileged_mask]

    if len(privileged_data) == 0 or len(unprivileged_data) == 0:
        logger.warning(f"Insufficient data for {metric_name.upper()} calculation: privileged={len(privileged_data)}, unprivileged={len(unprivileged_data)} samples. Returning NaN.")
        return MetricValueCarrier(float('nan'))

    # Calculate favorable outcome rates
    priv_favorable_count = len(privileged_data[privileged_data[outcome_name].isin(favorable_values)])
    unpriv_favorable_count = len(unprivileged_data[unprivileged_data[outcome_name].isin(favorable_values)])

    priv_favorable_rate = priv_favorable_count / len(privileged_data)
    unpriv_favorable_rate = unpriv_favorable_count / len(unprivileged_data)

    if metric_name.lower() == "spd":
        # SPD = P(Y=favorable|A=unprivileged) - P(Y=favorable|A=privileged)
        spd_value = unpriv_favorable_rate - priv_favorable_rate
        logger.debug(f"SPD calculation: privileged_rate={priv_favorable_rate:.4f}, unprivileged_rate={unpriv_favorable_rate:.4f}, spd={spd_value:.4f}")
        return MetricValueCarrier(spd_value)
    elif metric_name.lower() == "dir":
        if priv_favorable_rate == 0:
            logger.warning(f"Privileged favorable rate is 0, cannot calculate DIR. Returning NaN.")
            return MetricValueCarrier(float('nan'))
        # DIR = P(Y=favorable|A=unprivileged) / P(Y=favorable|A=privileged)
        dir_value = unpriv_favorable_rate / priv_favorable_rate
        logger.debug(f"DIR calculation: privileged_rate={priv_favorable_rate:.4f}, unprivileged_rate={unpriv_favorable_rate:.4f}, dir={dir_value:.4f}")
        return MetricValueCarrier(dir_value)
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")