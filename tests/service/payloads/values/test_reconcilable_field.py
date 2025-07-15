import pytest
from src.service.payloads.values.data_type import DataType
from src.service.payloads.values.reconcilable_field import ReconcilableField
from src.service.payloads.values.typed_value import TypedValue


class TestReconcilableField:
    """Test ReconcilableField functionality."""

    def test_single_value_initialization(self):
        """Test initialization with single value node."""
        value_node = {"type": "DOUBLE", "value": 1.0}
        field = ReconcilableField(value_node)

        assert field.raw_value_node == value_node
        assert field.raw_value_nodes is None
        assert not field.is_multiple_valued()
        assert field.get_reconciled_type() is None

    def test_multiple_values_initialization(self):
        """Test initialization with multiple value nodes."""
        value_nodes = [
            {"type": "DOUBLE", "value": 1.0},
            {"type": "DOUBLE", "value": 0.0},
        ]
        field = ReconcilableField(value_nodes)

        assert field.raw_value_nodes == value_nodes
        assert field.raw_value_node is None
        assert field.is_multiple_valued()

    def test_get_raw_value_nodes_single(self):
        """Test getting raw value nodes from single-valued field."""
        value_node = {"type": "DOUBLE", "value": 1.0}
        field = ReconcilableField(value_node)

        nodes = field.get_raw_value_nodes()
        assert len(nodes) == 1
        assert nodes[0] == value_node

    def test_get_raw_value_nodes_multiple(self):
        """Test getting raw value nodes from multi-valued field."""
        value_nodes = [
            {"type": "DOUBLE", "value": 1.0},
            {"type": "DOUBLE", "value": 0.0},
        ]
        field = ReconcilableField(value_nodes)

        nodes = field.get_raw_value_nodes()
        assert len(nodes) == 2
        assert nodes == value_nodes

    def test_get_raw_value_node_single(self):
        """Test getting single raw value node."""
        value_node = {"type": "DOUBLE", "value": 1.0}
        field = ReconcilableField(value_node)

        node = field.get_raw_value_node()
        assert node == value_node

    def test_get_raw_value_node_multiple_raises_error(self):
        """Test that getting single node from multi-valued field raises error."""
        value_nodes = [
            {"type": "DOUBLE", "value": 1.0},
            {"type": "DOUBLE", "value": 0.0},
        ]
        field = ReconcilableField(value_nodes)

        with pytest.raises(
            ValueError,
            match="Cannot return single value of multiple-valued ReconcilableField",
        ):
            field.get_raw_value_node()

    def test_set_and_get_reconciled_type(self):
        """Test setting and getting reconciled type."""
        field = ReconcilableField({"type": "DOUBLE", "value": 1.0})

        # Initially None
        assert field.get_reconciled_type() is None

        # Create typed values
        typed_value = TypedValue()
        typed_value.set_type(DataType.DOUBLE)
        typed_value.set_value({"value": 1.0})
        reconciled_types = [typed_value]

        # Set reconciled type
        field.set_reconciled_type(reconciled_types)
        assert field.get_reconciled_type() == reconciled_types
        assert len(field.get_reconciled_type()) == 1
        assert field.get_reconciled_type()[0].get_type() == DataType.DOUBLE

    def test_str_representation(self):
        """Test string representation of reconcilable field."""
        # Single value
        single_field = ReconcilableField({"type": "DOUBLE", "value": 1.0})
        assert str(single_field) == "{'type': 'DOUBLE', 'value': 1.0}"

        # Multiple values
        multiple_field = ReconcilableField([{"type": "DOUBLE", "value": 1.0}])
        assert str(multiple_field) == "[{'type': 'DOUBLE', 'value': 1.0}]"
