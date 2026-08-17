"""Tests for feature flag configuration."""

import importlib
from unittest.mock import patch

import pytest

from trustyai_service.service.config import feature_flags

_ALL_FLAGS = frozenset(
    {
        "fairness",
        "fairness_spd",
        "fairness_dir",
        "drift",
        "drift_ks_test",
        "drift_jensen_shannon",
        "drift_compare_means",
        "drift_mmd",
        "explainer",
        "explainer_local",
        "explainer_global",
    }
)


class TestFlagDefaults:
    """Verify default values for all feature flags."""

    def test_fairness_enabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["fairness"] is True

    def test_fairness_spd_enabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["fairness_spd"] is True

    def test_fairness_dir_enabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["fairness_dir"] is True

    def test_drift_enabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["drift"] is True

    def test_drift_ks_test_enabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["drift_ks_test"] is True

    def test_drift_jensen_shannon_enabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["drift_jensen_shannon"] is True

    def test_drift_compare_means_enabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["drift_compare_means"] is True

    def test_drift_mmd_enabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["drift_mmd"] is True

    def test_explainer_disabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["explainer"] is False

    def test_explainer_local_disabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["explainer_local"] is False

    def test_explainer_global_disabled_by_default(self) -> None:
        assert feature_flags.ENDPOINTS["explainer_global"] is False


class TestEnvironmentOverrides:
    """Verify environment variable overrides for feature flags."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled"])
    def test_truthy_values_enable_flag(self, value: str) -> None:
        with patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": value}):
            importlib.reload(feature_flags)
            assert feature_flags.ENDPOINTS["drift"] is True
        importlib.reload(feature_flags)

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled"])
    def test_falsy_values_disable_flag(self, value: str) -> None:
        with patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": value}):
            importlib.reload(feature_flags)
            assert feature_flags.ENDPOINTS["drift"] is False
        importlib.reload(feature_flags)

    def test_case_insensitive(self) -> None:
        with patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "TRUE"}):
            importlib.reload(feature_flags)
            assert feature_flags.ENDPOINTS["drift"] is True
        importlib.reload(feature_flags)

    def test_whitespace_stripped(self) -> None:
        with patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "  false  "}):
            importlib.reload(feature_flags)
            assert feature_flags.ENDPOINTS["drift"] is False
        importlib.reload(feature_flags)

    def test_unrecognized_value_keeps_default(self) -> None:
        with patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "maybe"}):
            importlib.reload(feature_flags)
            assert feature_flags.ENDPOINTS["drift"] is True
        importlib.reload(feature_flags)

    def test_enable_explainer(self) -> None:
        with patch.dict("os.environ", {"TRUSTYAI_ENABLE_EXPLAINER": "true"}):
            importlib.reload(feature_flags)
            assert feature_flags.ENDPOINTS["explainer"] is True
        importlib.reload(feature_flags)
