from src.service.payloads.values.reconcilable_field import ReconcilableField

class ReconcilableOutput(ReconcilableField):
    """
    Class for reconcilable output fields.
    """
    def __init__(self, raw_value_node) -> None:
        super().__init__(raw_value_node)