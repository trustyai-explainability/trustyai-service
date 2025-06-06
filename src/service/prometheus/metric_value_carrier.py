from typing import Dict, Optional, Union

from src.service.utils.exceptions import UnsupportedOperationException


class MetricValueCarrier:
    value: Optional[float]
    named_values: Optional[Dict[str, float]]
    single: bool

    def __init__(self, value_or_named_values: Union[float, Dict[str, float]]) -> None:
        if value_or_named_values is None or not isinstance(
            value_or_named_values, (int, float, dict)
        ):
            raise ValueError("Value must be a number or dictionary")

        if isinstance(value_or_named_values, (int, float)):
            self.value: float = float(value_or_named_values)
            self.named_values: Optional[Dict[str, float]] = None
            self.single: bool = True
        elif isinstance(value_or_named_values, dict):
            self.value: Optional[float] = None
            self.named_values: Dict[str, float] = value_or_named_values
            self.single: bool = False

    def is_single(self) -> bool:
        return self.single

    def get_value(self) -> float:
        if self.single:
            return self.value
        else:
            raise UnsupportedOperationException(
                "Metric value is not singular and therefore must be accessed via .get_value()"
            )

    def get_named_values(self) -> Dict[str, float]:
        if not self.single:
            return self.named_values
        else:
            raise UnsupportedOperationException(
                "Metric value is singular and therefore must be accessed via .get_named_values()"
            )
