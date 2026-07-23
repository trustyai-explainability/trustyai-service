"""Utility functions and models for group fairness metric endpoints."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trustyai_service.service.data.datasources.data_source import DataSource
from trustyai_service.service.data.shared_data_source import get_shared_data_source
from trustyai_service.service.payloads.metrics.base_metric_request import (
    BaseMetricRequest,
)
from trustyai_service.service.prometheus.prometheus_scheduler import PrometheusScheduler
from trustyai_service.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)

logger = logging.getLogger(__name__)


class ReconcilableFeature(BaseModel):
    """Reconcilable feature value representation for fairness metrics."""

    rawValueNodes: list[dict[str, Any]] | None = None
    rawValueNode: dict[str, Any] | None = None
    reconciledType: list[dict[str, Any]] | None = None
    multipleValued: bool | None = None


class ReconcilableOutput(BaseModel):
    """Reconcilable output value representation for fairness metrics."""

    rawValueNodes: list[dict[str, Any]] | None = None
    rawValueNode: dict[str, Any] | None = None
    reconciledType: list[dict[str, Any]] | None = None
    multipleValued: bool | None = None


class GroupMetricRequest(BaseMetricRequest):
    """Request parameters for group fairness metric computation."""

    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_id: str = Field(alias="modelId")
    metric_name: str | None = Field(
        default=None, alias="metricName"
    )  # Will be set by endpoint
    request_name: str | None = Field(default=None, alias="requestName")
    batch_size: int | None = Field(default=100, alias="batchSize")

    # SPD-specific fields
    protected_attribute: str = Field(alias="protectedAttribute")
    outcome_name: str = Field(alias="outcomeName")
    privileged_attribute: ReconcilableFeature | int | float | str = Field(
        alias="privilegedAttribute"
    )
    unprivileged_attribute: ReconcilableFeature | int | float | str = Field(
        alias="unprivilegedAttribute"
    )
    favorable_outcome: ReconcilableOutput | int | float | str = Field(
        alias="favorableOutcome"
    )
    threshold_delta: float | None = Field(default=None, alias="thresholdDelta")

    def retrieve_tags(self) -> dict[str, str]:
        """Retrieve tags for this SPD metric request."""
        tags = self.retrieve_default_tags()
        tags["protectedAttribute"] = self.protected_attribute
        tags["outcomeName"] = self.outcome_name
        return tags


class GroupDefinitionRequest(BaseModel):
    """Request payload for defining fairness groups with camelCase fields."""

    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    request_name: str | None = Field(default=None, alias="requestName")
    metric_name: str | None = Field(default=None, alias="metricName")
    batch_size: int | None = Field(default=100, alias="batchSize")
    protected_attribute: str = Field(alias="protectedAttribute")
    outcome_name: str = Field(alias="outcomeName")
    privileged_attribute: ReconcilableFeature | int | float | str = Field(
        alias="privilegedAttribute"
    )
    unprivileged_attribute: ReconcilableFeature | int | float | str = Field(
        alias="unprivilegedAttribute"
    )
    favorable_outcome: ReconcilableOutput | int | float | str = Field(
        alias="favorableOutcome"
    )
    threshold_delta: float | None = Field(default=None, alias="thresholdDelta")
    metric_value: dict[str, Any] = Field(alias="metricValue")


class ScheduleId(BaseModel):
    """Identifier for a scheduled metric computation request."""

    requestId: str


def get_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


def get_data_source() -> DataSource:
    """Get the shared data source instance."""
    return get_shared_data_source()


def _extract_values(
    value: ReconcilableFeature | ReconcilableOutput | float | str | list[Any] | None,
) -> list[Any]:
    """Extract raw values from Reconcilable* objects or primitives.

    Handles multiple value representations:
    - reconciledType (list of dicts with "value" key)
    - rawValueNodes (list of dicts with "value" key)
    - rawValueNode (single dict with "value" key)
    - Primitives (int, float, str)
    - Lists of primitives

    Args:
        value: Value to extract from (Reconcilable object, primitive, or list)

    Returns:
        List of extracted primitive values

    """
    if value is None:
        return []

    # Handle reconciledType (already processed values)
    if hasattr(value, "reconciledType") and value.reconciledType is not None:
        return [item["value"] for item in value.reconciledType if "value" in item]

    # Handle rawValueNodes (list of raw value nodes)
    if hasattr(value, "rawValueNodes") and value.rawValueNodes is not None:
        return [node["value"] for node in value.rawValueNodes if "value" in node]

    # Handle rawValueNode (single raw value node)
    if (
        hasattr(value, "rawValueNode")
        and value.rawValueNode is not None
        and "value" in value.rawValueNode
    ):
        return [value.rawValueNode["value"]]

    # Reject Reconcilable* objects that had no usable values
    if isinstance(value, (ReconcilableFeature, ReconcilableOutput)):
        # All known value-bearing attributes were None/empty
        return []

    # Handle primitives and lists
    return value if isinstance(value, list) else [value]


def prepare_fairness_data(
    dataframe: pd.DataFrame, request: GroupMetricRequest
) -> tuple[pd.DataFrame, pd.DataFrame, str, np.ndarray]:
    """Prepare data for fairness metric calculation by filtering into privileged and unprivileged groups.

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
    # Extract attribute names (both request types now expose snake_case attributes)
    protected_attr = request.protected_attribute
    outcome_name = request.outcome_name

    # Extract privileged/unprivileged group values from request
    privileged_values = _extract_values(request.privileged_attribute)
    unprivileged_values = _extract_values(request.unprivileged_attribute)

    # Extract favorable outcome values from request
    favorable_values = _extract_values(request.favorable_outcome)

    # Validate extracted values are non-empty and scalar-compatible
    if not privileged_values or any(
        isinstance(v, (BaseModel, dict, list)) for v in privileged_values
    ):
        msg = "privilegedAttribute must contain at least one scalar value"
        raise ValueError(msg)
    if not unprivileged_values or any(
        isinstance(v, (BaseModel, dict, list)) for v in unprivileged_values
    ):
        msg = "unprivilegedAttribute must contain at least one scalar value"
        raise ValueError(msg)
    if not favorable_values or any(
        isinstance(v, (BaseModel, dict, list)) for v in favorable_values
    ):
        msg = "favorableOutcome must contain at least one scalar value"
        raise ValueError(msg)

    # Filter the dataframe into privileged and unprivileged groups
    privileged_mask = dataframe[protected_attr].isin(privileged_values)
    unprivileged_mask = dataframe[protected_attr].isin(unprivileged_values)

    privileged_data = dataframe[privileged_mask]
    unprivileged_data = dataframe[unprivileged_mask]

    return privileged_data, unprivileged_data, outcome_name, np.array(favorable_values)
