import pytest
from src.service.prometheus.metric_value_carrier import (
    MetricValueCarrier,
    UnsupportedOperationException,
)


class TestMetricValueCarrier:
    def test_single_value_construction(self):
        """Test creation with single float value."""
        carrier = MetricValueCarrier(0.5)

        assert carrier.is_single() is True
        assert carrier.get_value() == 0.5

        with pytest.raises(
            UnsupportedOperationException,
            match="must be accessed via .get_named_values()",
        ):
            carrier.get_named_values()

    def test_single_value_construction_with_int(self):
        """Test creation with single int value (should convert to float)."""
        carrier = MetricValueCarrier(1)

        assert carrier.is_single() is True
        assert carrier.get_value() == 1.0
        assert isinstance(carrier.get_value(), float)

    def test_named_values_construction(self):
        """Test creation with named values dictionary."""
        named_values = {"feature1": 0.3, "feature2": 0.7}
        carrier = MetricValueCarrier(named_values)

        assert carrier.is_single() is False
        assert carrier.get_named_values() == named_values

        with pytest.raises(UnsupportedOperationException, match="must be accessed via .get_value()"):
            carrier.get_value()

    def test_empty_named_values(self):
        """Test creation with empty named values dictionary."""
        carrier = MetricValueCarrier({})

        assert carrier.is_single() is False
        assert carrier.get_named_values() == {}

    def test_invalid_construction(self):
        """Test that invalid inputs raise appropriate errors."""
        with pytest.raises(ValueError, match="Value must be a number or dictionary"):
            MetricValueCarrier("invalid")

        with pytest.raises(ValueError, match="Value must be a number or dictionary"):
            MetricValueCarrier(None)

        with pytest.raises(ValueError, match="Value must be a number or dictionary"):
            MetricValueCarrier([1, 2, 3])
