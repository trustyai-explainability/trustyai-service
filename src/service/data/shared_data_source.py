"""
Shared DataSource singleton to ensure consumer and scheduler use the same instance.
"""

from src.service.data.datasources.data_source import DataSource

# Global shared DataSource instance
_shared_data_source = None


def get_shared_data_source() -> DataSource:
    """
    Get the shared DataSource instance used by both consumer and scheduler.

    Returns:
        The singleton DataSource instance
    """
    global _shared_data_source
    if _shared_data_source is None:
        _shared_data_source = DataSource()
    return _shared_data_source
