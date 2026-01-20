"""
Shared PrometheusScheduler singleton to ensure endpoints and background scheduler use the same instance.
"""

from src.service.prometheus.prometheus_scheduler import PrometheusScheduler

# Global shared PrometheusScheduler instance
_shared_prometheus_scheduler = None


def get_shared_prometheus_scheduler() -> PrometheusScheduler:
    """
    Get the shared PrometheusScheduler instance used by both endpoints and background scheduler.

    Returns:
        The singleton PrometheusScheduler instance
    """
    global _shared_prometheus_scheduler
    if _shared_prometheus_scheduler is None:
        _shared_prometheus_scheduler = PrometheusScheduler()
    return _shared_prometheus_scheduler
