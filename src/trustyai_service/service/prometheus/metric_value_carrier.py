"""Container for metric values to be published to Prometheus."""

from typing import cast

from trustyai_service.service.utils.exceptions import UnsupportedOperationError


class MetricValueCarrier:
    """Carries metric values for Prometheus publication, supporting single or multi-valued metrics."""

    value: float | None
    named_values: dict[str, float] | None
    single: bool

    def __init__(self, value_or_named_values: float | dict[str, float]) -> None:
        """Initialize metric value carrier.

        :param value_or_named_values: Single numeric value or dictionary of named values
        :raises ValueError: If value is not a number or dictionary
        """
        if value_or_named_values is None or not isinstance(
            value_or_named_values, (int, float, dict)
        ):
            msg = "Value must be a number or dictionary"
            raise ValueError(msg)

        if isinstance(value_or_named_values, (int, float)):
            self.value: float = float(value_or_named_values)
            self.named_values: dict[str, float] | None = None
            self.single: bool = True
        elif isinstance(value_or_named_values, dict):
            self.value: float | None = None
            self.named_values: dict[str, float] = value_or_named_values
            self.single: bool = False

    def is_single(self) -> bool:
        """Check if this carrier contains a single value."""
        return self.single

    def get_value(self) -> float:
        """Get the single metric value.

        :return: The metric value
        :raises UnsupportedOperationError: If this carrier contains named values
        """
        if self.single:
            # Safe: __init__ guarantees value is not None when single=True
            return cast("float", self.value)
        msg = "Metric value is not singular and therefore must be accessed via .get_value()"
        raise UnsupportedOperationError(msg)

    def get_named_values(self) -> dict[str, float]:
        """Get the named metric values.

        :return: Dictionary of named metric values
        :raises UnsupportedOperationError: If this carrier contains a single value
        """
        if not self.single:
            # Safe: __init__ guarantees named_values is not None when single=False
            return cast("dict[str, float]", self.named_values)
        msg = "Metric value is singular and therefore must be accessed via .get_named_values()"
        raise UnsupportedOperationError(msg)
