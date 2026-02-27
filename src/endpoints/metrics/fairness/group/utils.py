import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.service.data.shared_data_source import get_shared_data_source
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.prometheus.shared_prometheus_scheduler import get_shared_prometheus_scheduler

logger = logging.getLogger(__name__)


class ReconcilableFeature(BaseModel):
    rawValueNodes: list[dict[str, Any]] | None = None
    rawValueNode: dict[str, Any] | None = None
    reconciledType: list[dict[str, Any]] | None = None
    multipleValued: bool | None = None


class ReconcilableOutput(BaseModel):
    rawValueNodes: list[dict[str, Any]] | None = None
    rawValueNode: dict[str, Any] | None = None
    reconciledType: list[dict[str, Any]] | None = None
    multipleValued: bool | None = None


class GroupMetricRequest(BaseMetricRequest):
    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_id: str = Field(alias="modelId")
    metric_name: str | None = Field(default=None, alias="metricName")  # Will be set by endpoint
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int | None = Field(default=100, alias="batchSize")

    # SPD-specific fields
    protected_attribute: str = Field(alias="protectedAttribute")
    outcome_name: str = Field(alias="outcomeName")
    privileged_attribute: ReconcilableFeature | int | float | str = Field(alias="privilegedAttribute")
    unprivileged_attribute: ReconcilableFeature | int | float | str = Field(alias="unprivilegedAttribute")
    favorable_outcome: ReconcilableOutput | int | float | str = Field(alias="favorableOutcome")
    threshold_delta: float | None = Field(default=None, alias="thresholdDelta")

    def retrieve_tags(self) -> dict[str, str]:
        """Retrieve tags for this SPD metric request."""
        tags = self.retrieve_default_tags()
        tags["protectedAttribute"] = self.protected_attribute
        tags["outcomeName"] = self.outcome_name
        return tags


class GroupDefinitionRequest(BaseModel):
    modelId: str
    requestName: str | None = None
    metricName: str | None = None
    batchSize: int | None = 100
    protectedAttribute: str
    outcomeName: str
    privilegedAttribute: ReconcilableFeature | int | float | str
    unprivilegedAttribute: ReconcilableFeature | int | float | str
    favorableOutcome: ReconcilableOutput | int | float | str
    thresholdDelta: float | None = None
    metricValue: dict[str, Any]

    # Additional snake_case properties for metrics processing
    @property
    def protected_attribute(self) -> str:
        return self.protectedAttribute

    @property
    def outcome_name(self) -> str:
        return self.outcomeName

    @property
    def privileged_attribute(self) -> ReconcilableFeature | int | float | str:
        return self.privilegedAttribute

    @property
    def unprivileged_attribute(self) -> ReconcilableFeature | int | float | str:
        return self.unprivilegedAttribute

    @property
    def favorable_outcome(self) -> ReconcilableOutput | int | float | str:
        return self.favorableOutcome

    @property
    def threshold_delta(self) -> float | None:
        return self.thresholdDelta


class ScheduleId(BaseModel):
    requestId: str


def get_prometheus_scheduler():
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source():
    """Get the shared data source instance."""
    return get_shared_data_source()


def prepare_fairness_data(
    dataframe: pd.DataFrame, request: GroupMetricRequest
) -> tuple[pd.DataFrame, pd.DataFrame, str, np.ndarray]:
    """
    Prepare data for fairness metric calculation by filtering into privileged and unprivileged groups.

    This function separates the input dataframe into two demographic groups based on a protected attribute
    (e.g., gender, race, age) to measure fairness in terms of how often each group receives a favorable outcome.

    Args:
        dataframe: The input dataframe containing both protected attributes and outcome predictions
        request: The fairness metric request specifying:
            - protectedAttribute: The demographic feature to check for bias (e.g., 'gender', 'race')
            - privilegedAttribute: Value(s) identifying the privileged demographic (e.g., 'male')
            - unprivilegedAttribute: Value(s) identifying the unprivileged demographic (e.g., 'female')
            - outcomeName: The prediction/outcome column (e.g., 'loan_decision')
            - favorableOutcome: The positive class value in outcomes (e.g., 'approved')

    Returns:
        Tuple of (privileged_data, unprivileged_data, outcome_name, favorable_values) where:
            - privileged_data: DataFrame subset for the privileged demographic group
            - unprivileged_data: DataFrame subset for the unprivileged demographic group
            - outcome_name: Name of the outcome column
            - favorable_values: NumPy array of outcome values considered favorable/positive (e.g., ['approved'], [1, 2])

    Example:
        For measuring loan approval fairness by gender:
            - privileged_data: All rows where gender='male'
            - unprivileged_data: All rows where gender='female'
            - outcome_name: 'loan_decision'
            - favorable_values: ['approved'] (could also be ['approved', 'conditionally_approved'])

        The fairness metrics (SPD/DIR) then compare:
            P(loan_decision in favorable_values | gender='female') vs P(loan_decision in favorable_values | gender='male')
    """
    # Extract attribute names
    protected_attr = (
        request.protectedAttribute if hasattr(request, "protectedAttribute") else request.protected_attribute
    )
    outcome_name = request.outcomeName if hasattr(request, "outcomeName") else request.outcome_name

    # Extract privileged/unprivileged group values from request
    if hasattr(request, "privilegedAttribute") and hasattr(request.privilegedAttribute, "reconciledType"):
        privileged_values = [
            item.get("value") for item in request.privilegedAttribute.reconciledType if "value" in item
        ]
    else:
        privileged_attr = getattr(request, "privilegedAttribute", getattr(request, "privileged_attribute", None))
        privileged_values = [privileged_attr] if not isinstance(privileged_attr, list) else privileged_attr

    if hasattr(request, "unprivilegedAttribute") and hasattr(request.unprivilegedAttribute, "reconciledType"):
        unprivileged_values = [
            item.get("value") for item in request.unprivilegedAttribute.reconciledType if "value" in item
        ]
    else:
        unprivileged_attr = getattr(request, "unprivilegedAttribute", getattr(request, "unprivileged_attribute", None))
        unprivileged_values = [unprivileged_attr] if not isinstance(unprivileged_attr, list) else unprivileged_attr

    # Extract favorable outcome values from request
    if hasattr(request, "favorableOutcome") and hasattr(request.favorableOutcome, "reconciledType"):
        favorable_values = [item.get("value") for item in request.favorableOutcome.reconciledType if "value" in item]
    else:
        favorable_attr = getattr(request, "favorableOutcome", getattr(request, "favorable_outcome", None))
        favorable_values = [favorable_attr] if not isinstance(favorable_attr, list) else favorable_attr

    # Filter the dataframe into privileged and unprivileged groups
    privileged_mask = dataframe[protected_attr].isin(privileged_values)
    unprivileged_mask = dataframe[protected_attr].isin(unprivileged_values)

    privileged_data = dataframe[privileged_mask]
    unprivileged_data = dataframe[unprivileged_mask]

    return privileged_data, unprivileged_data, outcome_name, np.array(favorable_values)
