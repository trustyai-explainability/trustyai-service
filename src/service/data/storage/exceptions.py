"""Custom exceptions for storage layer operations.

These exceptions allow callers to distinguish between different types of
failures and handle them appropriately.
"""


class StorageError(Exception):
    """Base exception for all storage-related errors."""


class DeserializationError(StorageError):
    """Raised when payload deserialization fails.

    This indicates corrupted data, unsupported format, or schema validation
    failure. This is distinct from "not found" and represents a data integrity
    issue that should be investigated.
    """

    def __init__(
        self, payload_id: str, reason: str, original_exception: Exception | None = None
    ) -> None:
        """Initialize deserialization error.

        Args:
            payload_id: The ID of the payload that failed to deserialize
            reason: Human-readable description of the failure
            original_exception: The underlying exception that caused the failure

        """
        self.payload_id = payload_id
        self.reason = reason
        self.original_exception = original_exception

        message = f"Failed to deserialize payload '{payload_id}': {reason}"
        if original_exception:
            message += f" (caused by: {type(original_exception).__name__}: {original_exception})"

        super().__init__(message)

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        caused_by = (
            f", caused_by={type(self.original_exception).__name__}"
            if self.original_exception
            else ""
        )
        return (
            f"DeserializationError(payload_id={self.payload_id!r}, "
            f"reason={self.reason!r}{caused_by})"
        )
