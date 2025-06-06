from src.service.payloads.values.reconcilable_field import ReconcilableField
from src.service.payloads.values.reconcilable_output import ReconcilableOutput

class TestReconcilableOutput:
    """Test ReconcilableOutput functionality."""
    
    def test_reconcilable_output_inheritance(self):
        """Test that ReconcilableOutput inherits from ReconcilableField."""
        value_node = {"type": "DOUBLE", "value": 1.0}
        output = ReconcilableOutput(value_node)
        
        assert isinstance(output, ReconcilableField)
        assert output.get_raw_value_node() == value_node
        assert not output.is_multiple_valued()