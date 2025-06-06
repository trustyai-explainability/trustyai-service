
class ReconcilerMatcher:
    """
    Decorator class for fields that need reconciliation.
    Use to mark a field as requiring reconciliation with a specific type.
    """
    def __init__(self, name_provider: str) -> None:
        if name_provider is None:
            raise ValueError("name_provider cannot be None")
        self.name_provider = name_provider
        
    def __call__(self, field_class):
        """Mark a field descriptor as needing reconciliation"""
        field_class._reconciler_matcher = self
        return field_class
