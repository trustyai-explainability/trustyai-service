"""Shared DataSource singleton to ensure consumer and scheduler use the same instance."""

import threading

from trustyai_service.service.data.datasources.data_source import DataSource

_state: dict[str, DataSource | None] = {"instance": None}
_lock = threading.Lock()


def get_shared_data_source() -> DataSource:
    """Get the shared DataSource instance used by both consumer and scheduler.

    Returns:
        The singleton DataSource instance

    """
    with _lock:
        if _state["instance"] is None:
            _state["instance"] = DataSource()
        return _state["instance"]


def reset_shared_data_source() -> None:
    """Reset singleton instance (useful for testing)."""
    with _lock:
        _state["instance"] = None
