import pytest
from src.service.payloads.values.reconciler_matcher import ReconcilerMatcher


class TestReconcilerMatcher:
    """Test ReconcilerMatcher functionality."""
    
    def test_reconciler_matcher_initialization(self):
        """Test basic initialization of ReconcilerMatcher."""
        matcher = ReconcilerMatcher("get_protected_attribute")
        assert matcher.name_provider == "get_protected_attribute"
    
    def test_reconciler_matcher_decorator_functionality(self):
        """Test that ReconcilerMatcher works as a decorator."""
        matcher = ReconcilerMatcher("get_field_name")
        
        @matcher
        class TestField:
            pass
        
        # Verify the decorator attached the matcher
        assert hasattr(TestField, '_reconciler_matcher')
        assert TestField._reconciler_matcher == matcher
        assert TestField._reconciler_matcher.name_provider == "get_field_name"
    
    def test_multiple_reconciler_matchers(self):
        """Test that multiple fields can have different matchers."""
        matcher1 = ReconcilerMatcher("get_input_name")
        matcher2 = ReconcilerMatcher("get_output_name")
        
        @matcher1
        class InputField:
            pass
        
        @matcher2
        class OutputField:
            pass
        
        assert InputField._reconciler_matcher.name_provider == "get_input_name"
        assert OutputField._reconciler_matcher.name_provider == "get_output_name"
        assert InputField._reconciler_matcher != OutputField._reconciler_matcher
    
    def test_reconciler_matcher_with_none_name_provider(self):
        """Test ReconcilerMatcher with None name provider."""
        with pytest.raises(ValueError):
            ReconcilerMatcher(None)