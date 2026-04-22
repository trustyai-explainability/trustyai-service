"""Feature flags controlling endpoint availability.

Disabled endpoints are excluded from the FastAPI router registry at startup.
Each flag can be overridden at deployment time via the environment variable
``TRUSTYAI_ENABLE_<FLAG_NAME>`` (upper-cased), e.g.
``TRUSTYAI_ENABLE_DRIFT=false`` disables the entire drift group.
"""

import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSY = frozenset({"0", "false", "no", "off", "disabled"})


def _flag(name: str, default: bool) -> bool:
    """Read a feature flag, allowing env-var override."""
    env = os.getenv(f"TRUSTYAI_ENABLE_{name.upper()}")
    if env is not None:
        normalized = env.strip().lower()
        if normalized in _TRUTHY:
            return True
        if normalized in _FALSY:
            return False
        logger.warning(
            "Ignoring unrecognized value %r for TRUSTYAI_ENABLE_%s; using default %s",
            env,
            name.upper(),
            default,
        )
    return default


ENDPOINTS: dict[str, bool] = {
    "fairness": _flag("fairness", True),
    "fairness_spd": _flag("fairness_spd", True),
    "fairness_dir": _flag("fairness_dir", True),
    "drift": _flag("drift", True),
    "drift_ks_test": _flag("drift_ks_test", True),
    "drift_jensen_shannon": _flag("drift_jensen_shannon", True),
    "drift_compare_means": _flag("drift_compare_means", True),
    "moving_average": _flag("moving_average", True),
    "data_download": _flag("data_download", False),
    "explainer": _flag("explainer", False),
    "explainer_local": _flag("explainer_local", False),
    "explainer_global": _flag("explainer_global", False),
}
