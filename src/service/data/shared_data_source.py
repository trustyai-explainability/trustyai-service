"""Shared DataSource singleton to ensure consumer and scheduler use the same.

instance.
"""

from src.service.data.datasources.data_source import DataSource


class SharedDataSource:
    """Singleton holder for shared DataSource instance."""

    _instance: DataSource | None = None

    @classmethod
    def get(cls) -> DataSource:
        """Get the shared DataSource instance used by both consumer and scheduler.

        Returns:
            The singleton DataSource instance

        """
        if cls._instance is None:
            cls._instance = DataSource()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None


def get_shared_data_source() -> DataSource:
    """Get the shared DataSource instance used by both consumer and scheduler.

    Returns:
        The singleton DataSource instance

    """
    return SharedDataSource.get()
