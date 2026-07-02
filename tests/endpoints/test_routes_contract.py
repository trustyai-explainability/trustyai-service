"""Contract tests for centralized endpoint path constants.

Prevents accidental path value changes that would break the external API.
"""

from src.endpoints import routes


class TestDriftPaths:
    """Drift metric paths match the API contract."""

    def test_compare_means(self) -> None:
        """CompareMeans paths."""
        assert routes.DRIFT_COMPARE_MEANS.compute == "/metrics/drift/comparemeans"
        assert (
            routes.DRIFT_COMPARE_MEANS.definition
            == "/metrics/drift/comparemeans/definition"
        )
        assert (
            routes.DRIFT_COMPARE_MEANS.request == "/metrics/drift/comparemeans/request"
        )
        assert (
            routes.DRIFT_COMPARE_MEANS.requests
            == "/metrics/drift/comparemeans/requests"
        )

    def test_meanshift_deprecated(self) -> None:
        """Meanshift is a deprecated alias for CompareMeans."""
        assert routes.DRIFT_MEANSHIFT.compute == "/metrics/drift/meanshift"

    def test_kstest(self) -> None:
        """KSTest paths."""
        assert routes.DRIFT_KSTEST.compute == "/metrics/drift/kstest"

    def test_jensen_shannon(self) -> None:
        """JensenShannon paths."""
        assert routes.DRIFT_JENSEN_SHANNON.compute == "/metrics/drift/jensenshannon"

    def test_fourier_mmd(self) -> None:
        """FourierMMD paths."""
        assert routes.DRIFT_FOURIER_MMD.compute == "/metrics/drift/fouriermmd"

    def test_approx_ks_test(self) -> None:
        """ApproxKSTest paths."""
        assert routes.DRIFT_APPROX_KS_TEST.compute == "/metrics/drift/approxkstest"


class TestFairnessPaths:
    """Fairness metric paths match the API contract."""

    def test_dir(self) -> None:
        """DIR paths."""
        assert routes.FAIRNESS_DIR.compute == "/metrics/group/fairness/dir"

    def test_spd(self) -> None:
        """SPD paths."""
        assert routes.FAIRNESS_SPD.compute == "/metrics/group/fairness/spd"

    def test_legacy_dir(self) -> None:
        """Legacy DIR deprecated alias."""
        assert routes.LEGACY_DIR.compute == "/dir"

    def test_legacy_spd(self) -> None:
        """Legacy SPD deprecated alias."""
        assert routes.LEGACY_SPD.compute == "/spd"


class TestOtherPaths:
    """Non-metric paths match the API contract."""

    def test_batch_mean(self) -> None:
        """BatchMean paths."""
        assert routes.BATCH_MEAN.compute == "/metrics/batchmean"

    def test_identity_deprecated(self) -> None:
        """Identity is a deprecated alias for BatchMean."""
        assert routes.IDENTITY.compute == "/metrics/identity"

    def test_info(self) -> None:
        """Service metadata paths."""
        assert routes.INFO == "/info"
        assert routes.INFO_NAMES == "/info/names"
        assert routes.INFO_TAGS == "/info/tags"
        assert routes.INFO_INFERENCE_IDS == "/info/inference/ids/{model}"

    def test_consumer(self) -> None:
        """Consumer paths."""
        assert routes.CONSUMER_ROOT == "/"
        assert routes.CONSUMER_KSERVE_V2 == "/consumer/kserve/v2"

    def test_data(self) -> None:
        """Data upload path."""
        assert routes.DATA_UPLOAD == "/data/upload"

    def test_health(self) -> None:
        """Health and monitoring paths."""
        assert routes.HEALTH_READY == "/q/health/ready"
        assert routes.HEALTH_LIVE == "/q/health/live"
        assert routes.PROMETHEUS_METRICS == "/q/metrics"

    def test_metrics_info(self) -> None:
        """Metrics info path."""
        assert routes.METRICS_ALL_REQUESTS == "/metrics/all/requests"

    def test_explainers(self) -> None:
        """Explainer paths."""
        assert routes.EXPLAINER_GLOBAL_LIME == "/explainers/global/lime"
        assert routes.EXPLAINER_GLOBAL_PDP == "/explainers/global/pdp"
        assert routes.EXPLAINER_LOCAL_LIME == "/explainers/local/lime"
        assert routes.EXPLAINER_LOCAL_SHAP == "/explainers/local/shap"
        assert routes.EXPLAINER_LOCAL_CF == "/explainers/local/cf"
        assert routes.EXPLAINER_LOCAL_TSSALIENCY == "/explainers/local/tssaliency"
