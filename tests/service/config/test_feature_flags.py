"""Unit tests for feature_flags.py."""

from unittest.mock import patch

from src.service.config.feature_flags import ENDPOINTS, _flag


class TestEndpointFlags:
    def test_all_flags_are_boolean(self):
        for value in ENDPOINTS.values():
            assert isinstance(value, bool)

    def test_all_known_flags_present(self):
        expected = {
            "fairness",
            "fairness_spd",
            "fairness_dir",
            "drift",
            "drift_ks_test",
            "drift_jensen_shannon",
            "drift_compare_means",
            "explainer",
            "explainer_local",
            "explainer_global",
            "data_download",
        }
        assert set(ENDPOINTS.keys()) == expected


class TestFlagEnvOverride:
    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "false"})
    def test_env_disables_flag(self):
        assert _flag("drift", True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "true"})
    def test_env_enables_flag(self):
        assert _flag("drift", False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "1"})
    def test_env_truthy_1(self):
        assert _flag("drift", False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "0"})
    def test_env_falsy_0(self):
        assert _flag("drift", True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "yes"})
    def test_env_truthy_yes(self):
        assert _flag("drift", False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "no"})
    def test_env_falsy_no(self):
        assert _flag("drift", True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "on"})
    def test_env_truthy_on(self):
        assert _flag("drift", False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "off"})
    def test_env_falsy_off(self):
        assert _flag("drift", True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "enabled"})
    def test_env_truthy_enabled(self):
        assert _flag("drift", False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "disabled"})
    def test_env_falsy_disabled(self):
        assert _flag("drift", True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "TRUE"})
    def test_env_case_insensitive(self):
        assert _flag("drift", False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "  true  "})
    def test_env_strips_whitespace(self):
        assert _flag("drift", False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "maybe"})
    def test_env_unrecognized_falls_back_to_default(self):
        assert _flag("drift", True) is True
        assert _flag("drift", False) is False

    @patch.dict("os.environ", {}, clear=False)
    def test_no_env_uses_default(self):
        # Ensure TRUSTYAI_ENABLE_NONEXISTENT is not set
        import os

        os.environ.pop("TRUSTYAI_ENABLE_NONEXISTENT", None)
        assert _flag("nonexistent", True) is True
        assert _flag("nonexistent", False) is False
