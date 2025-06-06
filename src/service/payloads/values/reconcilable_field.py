from typing import List, Optional, Union, Dict, Any
from src.service.payloads.values.typed_value import TypedValue


class ReconcilableField:
    """
    Base class for reconcilable fields that need type reconciliation.
    """
    
    def __init__(self, raw_value_node: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Initialize with a raw value node or list of nodes.
        
        Args:
            raw_value_node: Single value node or list of value nodes
        """
        if isinstance(raw_value_node, list):
            self.raw_value_nodes = raw_value_node
            self.raw_value_node = None
        else:
            self.raw_value_node = raw_value_node
            self.raw_value_nodes = None
            
        self.reconciled_type: Optional[List[TypedValue]] = None
    
    def get_raw_value_nodes(self) -> List[Dict[str, Any]]:
        """
        Get the raw value nodes.
        
        Returns:
            A list of raw value nodes
        """
        if self.is_multiple_valued():
            return self.raw_value_nodes
        else:
            return [self.raw_value_node]
    
    def get_raw_value_node(self) -> Dict[str, Any]:
        """
        Get the single raw value node.
        
        Returns:
            The raw value node
            
        Raises:
            ValueError: If this field has multiple values
        """
        if not self.is_multiple_valued():
            return self.raw_value_node
        else:
            raise ValueError("Cannot return single value of multiple-valued ReconcilableField")
    
    def is_multiple_valued(self) -> bool:
        """
        Check if this field has multiple values.
        
        Returns:
            True if this field has multiple values, False otherwise
        """
        return self.raw_value_nodes is not None
    
    def get_reconciled_type(self) -> Optional[List[TypedValue]]:
        """
        Get the reconciled type information.
        
        Returns:
            The reconciled type information or None if not reconciled
        """
        return self.reconciled_type
    
    def set_reconciled_type(self, reconciled_type: List[TypedValue]) -> None:
        """
        Set the reconciled type information.
        
        Args:
            reconciled_type: The reconciled type information
        """
        self.reconciled_type = reconciled_type
    
    def __str__(self) -> str:
        if self.is_multiple_valued():
            return str(self.raw_value_nodes)
        else:
            return str(self.raw_value_node)
