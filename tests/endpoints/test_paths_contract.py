"""Contract tests for centralized endpoint path constants.

Prevents accidental path value changes that would break the external API.
"""

from src.endpoints.paths import (
    BATCH_MEAN,
    CONSUMER_KSERVE_V2,
    CONSUMER_ROOT,
    DATA_UPLOAD,
    DRIFT_APPROX_KS_TEST,
    DRIFT_COMPARE_MEANS,
    DRIFT_FOURIER_MMD,
    DRIFT_JENSEN_SHANNON,
    DRIFT_KSTEST,
    DRIFT_MEANSHIFT,
    FAIRNESS_DIR,
    FAIRNESS_SPD,
    HEALTH_LIVE,
    HEALTH_READY,
    IDENTITY,
    INFO,
    INFO_INFERENCE_IDS,
    INFO_NAMES,
    INFO_TAGS,
    METRICS_ALL_REQUESTS,
    PROMETHEUS_METRICS,
)


class TestDriftPaths:
    """Drift metric paths match the API contract."""

    def test_compare_means(self) -> None:
        """CompareMeans paths."""
        assert DRIFT_COMPARE_MEANS.compute == "/metrics/drift/comparemeans"
        assert (
            DRIFT_COMPARE_MEANS.definition == "/metrics/drift/comparemeans/definition"
        )
        assert DRIFT_COMPARE_MEANS.request == "/metrics/drift/comparemeans/request"
        assert DRIFT_COMPARE_MEANS.requests == "/metrics/drift/comparemeans/requests"

    def test_meanshift_deprecated(self) -> None:
        """Meanshift is a deprecated alias for CompareMeans."""
        assert DRIFT_MEANSHIFT.compute == "/metrics/drift/meanshift"

    def test_kstest(self) -> None:
        """KSTest paths."""
        assert DRIFT_KSTEST.compute == "/metrics/drift/kstest"

    def test_jensen_shannon(self) -> None:
        """JensenShannon paths."""
        assert DRIFT_JENSEN_SHANNON.compute == "/metrics/drift/jensenshannon"

    def test_fourier_mmd(self) -> None:
        """FourierMMD paths."""
        assert DRIFT_FOURIER_MMD.compute == "/metrics/drift/fouriermmd"

    def test_approx_ks_test(self) -> None:
        """ApproxKSTest paths."""
        assert DRIFT_APPROX_KS_TEST.compute == "/metrics/drift/approxkstest"


class TestFairnessPaths:
    """Fairness metric paths match the API contract."""

    def test_dir(self) -> None:
        """DIR paths."""
        assert FAIRNESS_DIR.compute == "/metrics/group/fairness/dir"

    def test_spd(self) -> None:
        """SPD paths."""
        assert FAIRNESS_SPD.compute == "/metrics/group/fairness/spd"


class TestOtherPaths:
    """Non-metric paths match the API contract."""

    def test_batch_mean(self) -> None:
        """BatchMean paths."""
        assert BATCH_MEAN.compute == "/metrics/batchmean"

    def test_identity_deprecated(self) -> None:
        """Identity is a deprecated alias for BatchMean."""
        assert IDENTITY.compute == "/metrics/identity"

    def test_info(self) -> None:
        """Service metadata paths."""
        assert INFO == "/info"
        assert INFO_NAMES == "/info/names"
        assert INFO_TAGS == "/info/tags"
        assert INFO_INFERENCE_IDS == "/info/inference/ids/{model}"

    def test_consumer(self) -> None:
        """Consumer paths."""
        assert CONSUMER_ROOT == "/"
        assert CONSUMER_KSERVE_V2 == "/consumer/kserve/v2"

    def test_data(self) -> None:
        """Data upload path."""
        assert DATA_UPLOAD == "/data/upload"

    def test_health(self) -> None:
        """Health and monitoring paths."""
        assert HEALTH_READY == "/q/health/ready"
        assert HEALTH_LIVE == "/q/health/live"
        assert PROMETHEUS_METRICS == "/q/metrics"

    def test_metrics_info(self) -> None:
        """Metrics info path."""
        assert METRICS_ALL_REQUESTS == "/metrics/all/requests"
