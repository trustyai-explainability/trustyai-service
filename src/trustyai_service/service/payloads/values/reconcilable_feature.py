"""Reconcilable feature field for input data type resolution."""

from typing import Any

from trustyai_service.service.payloads.values.reconcilable_field import (
    ReconcilableField,
)


class ReconcilableFeature(ReconcilableField):
    """Class for reconcilable feature fields."""

    def __init__(self, raw_value_node: dict[str, Any] | list[dict[str, Any]]) -> None:
        """Initialize reconcilable feature from raw value node.

        :param raw_value_node: Raw value data to reconcile
        """
        super().__init__(raw_value_node)
