"""Decorator for marking fields that require type reconciliation."""

from typing import TypeVar

T = TypeVar("T", bound=type)


class ReconcilerMatcher:
    """Decorator class for fields that need reconciliation.

    Use to mark a field as requiring reconciliation with a specific
    type.
    """

    def __init__(self, name_provider: str) -> None:
        """Initialize the reconciler matcher.

        :param name_provider: Name provider for the field
        :raises ValueError: If name_provider is None
        """
        if name_provider is None:
            msg = "name_provider cannot be None"
            raise ValueError(msg)
        self.name_provider = name_provider

    def __call__(self, field_class: T) -> T:
        """Mark a field descriptor as needing reconciliation."""
        # Set private attribute for reconciliation metadata
        field_class._reconciler_matcher = self  # type: ignore[attr-defined]
        return field_class
