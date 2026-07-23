"""Base class for reconcilable fields that need type reconciliation."""

from typing import Any, cast

from trustyai_service.service.payloads.values.typed_value import TypedValue


class ReconcilableField:
    """Base class for reconcilable fields that need type reconciliation."""

    def __init__(self, raw_value_node: dict[str, Any] | list[dict[str, Any]]) -> None:
        """Initialize with a raw value node or list of nodes.

        Args:
            raw_value_node: Single value node or list of value nodes

        """
        if isinstance(raw_value_node, list):
            self.raw_value_nodes: list[dict[str, Any]] | None = raw_value_node
            self.raw_value_node: dict[str, Any] | None = None
        else:
            self.raw_value_node = raw_value_node
            self.raw_value_nodes = None

        self.reconciled_type: list[TypedValue] | None = None

    def get_raw_value_nodes(self) -> list[dict[str, Any]]:
        """Get the raw value nodes.

        Returns:
            A list of raw value nodes

        """
        if self.is_multiple_valued():
            # Safe: __init__ guarantees raw_value_nodes is not None when is_multiple_valued()
            return cast("list[dict[str, Any]]", self.raw_value_nodes)
        # Safe: __init__ guarantees raw_value_node is not None when not is_multiple_valued()
        return [cast("dict[str, Any]", self.raw_value_node)]

    def get_raw_value_node(self) -> dict[str, Any]:
        """Get the single raw value node.

        Returns:
            The raw value node

        Raises:
            ValueError: If this field has multiple values

        """
        if not self.is_multiple_valued():
            # Safe: __init__ guarantees raw_value_node is not None when not is_multiple_valued()
            return cast("dict[str, Any]", self.raw_value_node)
        msg = "Cannot return single value of multiple-valued ReconcilableField"
        raise ValueError(
            msg,
        )

    def is_multiple_valued(self) -> bool:
        """Check if this field has multiple values.

        Returns:
            True if this field has multiple values, False otherwise

        """
        return self.raw_value_nodes is not None

    def get_reconciled_type(self) -> list[TypedValue] | None:
        """Get the reconciled type information.

        Returns:
            The reconciled type information or None if not reconciled

        """
        return self.reconciled_type

    def set_reconciled_type(self, reconciled_type: list[TypedValue]) -> None:
        """Set the reconciled type information.

        Args:
            reconciled_type: The reconciled type information

        """
        self.reconciled_type = reconciled_type

    def __str__(self) -> str:
        """Return string representation of the reconcilable field."""
        if self.is_multiple_valued():
            return str(self.raw_value_nodes)
        return str(self.raw_value_node)
