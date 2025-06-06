from src.service.payloads.values.data_type import DataType
from src.service.payloads.values.typed_value import TypedValue

class TestTypedValue:
    """Test TypedValue functionality."""
    
    def test_typed_value_initialization(self):
        """Test TypedValue initialization."""
        tv = TypedValue()
        assert tv.get_type() is None
        assert tv.get_value() is None
    
    def test_set_and_get_type(self):
        """Test setting and getting type."""
        tv = TypedValue()
        tv.set_type(DataType.DOUBLE)
        assert tv.get_type() == DataType.DOUBLE
    
    def test_set_and_get_value(self):
        """Test setting and getting value."""
        tv = TypedValue()
        value = {"numeric_value": 1.0}
        tv.set_value(value)
        assert tv.get_value() == value
    
    def test_complete_typed_value(self):
        """Test complete TypedValue setup."""
        tv = TypedValue()
        tv.set_type(DataType.STRING)
        tv.set_value({"string_value": "test"})
        
        assert tv.get_type() == DataType.STRING
        assert tv.get_value() == {"string_value": "test"}