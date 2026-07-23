"""Shared PrometheusScheduler singleton for endpoints and background scheduler."""

from trustyai_service.service.prometheus.prometheus_scheduler import PrometheusScheduler


class SharedPrometheusScheduler:
    """Singleton holder for shared PrometheusScheduler instance."""

    _instance: PrometheusScheduler | None = None

    @classmethod
    def get(cls) -> PrometheusScheduler:
        """Get the shared PrometheusScheduler instance used by both endpoints and background scheduler.

        Returns:
            The singleton PrometheusScheduler instance

        """
        if cls._instance is None:
            cls._instance = PrometheusScheduler()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None


def get_shared_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared PrometheusScheduler instance used by both endpoints and background scheduler.

    Returns:
        The singleton PrometheusScheduler instance

    """
    return SharedPrometheusScheduler.get()
