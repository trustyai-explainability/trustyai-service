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


def _flag(name: str, *, default: bool) -> bool:
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
    "fairness": _flag("fairness", default=True),
    "fairness_spd": _flag("fairness_spd", default=True),
    "fairness_dir": _flag("fairness_dir", default=True),
    "drift": _flag("drift", default=True),
    "drift_ks_test": _flag("drift_ks_test", default=True),
    "drift_jensen_shannon": _flag("drift_jensen_shannon", default=True),
    "drift_compare_means": _flag("drift_compare_means", default=True),
    "data_download": _flag("data_download", default=False),
    "explainer": _flag("explainer", default=False),
    "explainer_local": _flag("explainer_local", default=False),
    "explainer_global": _flag("explainer_global", default=False),
}
