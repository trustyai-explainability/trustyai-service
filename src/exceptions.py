"""Custom exceptions for TrustyAI service."""


class ReconciliationError(Exception):
    """
    Raised when payload reconciliation fails.

    This exception is raised when KServe inference payloads cannot be reconciled
    due to mismatched shapes, row counts, or other data inconsistencies.
    """

    def __init__(self, message: str, payload_id: str = None, model_id: str = None):
        """
        Initialize ReconciliationError.

        Args:
            message: Detailed error message
            payload_id: Optional ID of the payload that failed reconciliation
            model_id: Optional model ID associated with the payload
        """
        self.message = message
        self.payload_id = payload_id
        self.model_id = model_id
        super().__init__(self.message)

    def __str__(self):
        parts = [self.message]
        if self.payload_id:
            parts.append(f"(payload_id: {self.payload_id})")
        if self.model_id:
            parts.append(f"(model_id: {self.model_id})")
        return " ".join(parts)
