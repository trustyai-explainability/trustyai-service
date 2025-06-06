from src.service.payloads.values.reconcilable_field import ReconcilableField
from src.service.payloads.values.reconcilable_feature import ReconcilableFeature


class TestReconcilableFeature:
    """Test ReconcilableFeature functionality."""
    
    def test_reconcilable_feature_inheritance(self):
        """Test that ReconcilableFeature inherits from ReconcilableField."""
        value_node = {"type": "STRING", "value": "gender"}
        feature = ReconcilableFeature(value_node)
        
        assert isinstance(feature, ReconcilableField)
        assert feature.get_raw_value_node() == value_node
        assert not feature.is_multiple_valued()