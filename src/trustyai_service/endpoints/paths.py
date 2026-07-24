"""Centralized endpoint path constants.

Single source of truth for all API paths. Prevents hardcoded string
duplication across endpoint files and tests.
"""

from typing import NamedTuple


class MetricPaths(NamedTuple):
    """Path constants for a standard metric endpoint group."""

    compute: str
    definition: str
    request: str
    requests: str


def _metric(base: str) -> MetricPaths:
    """Generate the standard 4-path tuple for a metric base path."""
    return MetricPaths(
        compute=base,
        definition=f"{base}/definition",
        request=f"{base}/request",
        requests=f"{base}/requests",
    )


# === Drift Metrics ===

DRIFT_COMPARE_MEANS = _metric("/metrics/drift/comparemeans")
DRIFT_MEANSHIFT = _metric("/metrics/drift/meanshift")  # deprecated alias
DRIFT_KSTEST = _metric("/metrics/drift/kstest")
DRIFT_JENSEN_SHANNON = _metric("/metrics/drift/jensenshannon")
DRIFT_FOURIER_MMD = _metric("/metrics/drift/fouriermmd")
DRIFT_APPROX_KS_TEST = _metric("/metrics/drift/approxkstest")

# === Fairness Metrics ===

FAIRNESS_DIR = _metric("/metrics/group/fairness/dir")
FAIRNESS_SPD = _metric("/metrics/group/fairness/spd")
LEGACY_DIR = _metric("/dir")  # deprecated
LEGACY_SPD = _metric("/spd")  # deprecated

# === Batch Mean / Identity ===

BATCH_MEAN = _metric("/metrics/batchmean")
IDENTITY = _metric("/metrics/identity")  # deprecated

# === Metrics Info ===

METRICS_ALL_REQUESTS = "/metrics/all/requests"

# === Service Metadata ===

INFO = "/info"
INFO_INFERENCE_IDS = "/info/inference/ids/{model}"
INFO_NAMES = "/info/names"
INFO_TAGS = "/info/tags"

# === Data ===

DATA_UPLOAD = "/data/upload"

# === Consumer ===

CONSUMER_ROOT = "/"
CONSUMER_KSERVE_V2 = "/consumer/kserve/v2"

# === Explainers ===

EXPLAINER_GLOBAL_LIME = "/explainers/global/lime"
EXPLAINER_GLOBAL_PDP = "/explainers/global/pdp"
EXPLAINER_LOCAL_LIME = "/explainers/local/lime"
EXPLAINER_LOCAL_SHAP = "/explainers/local/shap"
EXPLAINER_LOCAL_CF = "/explainers/local/cf"
EXPLAINER_LOCAL_TSSALIENCY = "/explainers/local/tssaliency"

# === LM Evaluation ===

EVAL_PREFIX = "/eval/lm-evaluation-harness"

# === Health & Monitoring ===

HEALTH_READY = "/q/health/ready"
HEALTH_LIVE = "/q/health/live"
PROMETHEUS_METRICS = "/q/metrics"
