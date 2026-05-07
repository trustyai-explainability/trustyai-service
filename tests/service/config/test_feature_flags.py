"""Unit tests for feature_flags.py."""

import os
from unittest.mock import patch

from src.service.config.feature_flags import ENDPOINTS, _flag


class TestEndpointFlags:
    """Tests for the ENDPOINTS dictionary defaults."""

    def test_all_flags_are_boolean(self) -> None:
        """Every value in ENDPOINTS must be a bool."""
        for value in ENDPOINTS.values():
            assert isinstance(value, bool)

    def test_all_known_flags_present(self) -> None:
        """ENDPOINTS must contain exactly the expected set of flag keys."""
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
    """Tests for _flag() environment variable override logic."""

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "false"})
    def test_env_disables_flag(self) -> None:
        """Env var 'false' overrides default=True."""
        assert _flag("drift", default=True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "true"})
    def test_env_enables_flag(self) -> None:
        """Env var 'true' overrides default=False."""
        assert _flag("drift", default=False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "1"})
    def test_env_truthy_1(self) -> None:
        """Env var '1' is truthy."""
        assert _flag("drift", default=False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "0"})
    def test_env_falsy_0(self) -> None:
        """Env var '0' is falsy."""
        assert _flag("drift", default=True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "yes"})
    def test_env_truthy_yes(self) -> None:
        """Env var 'yes' is truthy."""
        assert _flag("drift", default=False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "no"})
    def test_env_falsy_no(self) -> None:
        """Env var 'no' is falsy."""
        assert _flag("drift", default=True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "on"})
    def test_env_truthy_on(self) -> None:
        """Env var 'on' is truthy."""
        assert _flag("drift", default=False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "off"})
    def test_env_falsy_off(self) -> None:
        """Env var 'off' is falsy."""
        assert _flag("drift", default=True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "enabled"})
    def test_env_truthy_enabled(self) -> None:
        """Env var 'enabled' is truthy."""
        assert _flag("drift", default=False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "disabled"})
    def test_env_falsy_disabled(self) -> None:
        """Env var 'disabled' is falsy."""
        assert _flag("drift", default=True) is False

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "TRUE"})
    def test_env_case_insensitive(self) -> None:
        """Env var parsing is case-insensitive."""
        assert _flag("drift", default=False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "  true  "})
    def test_env_strips_whitespace(self) -> None:
        """Env var parsing strips surrounding whitespace."""
        assert _flag("drift", default=False) is True

    @patch.dict("os.environ", {"TRUSTYAI_ENABLE_DRIFT": "maybe"})
    def test_env_unrecognized_falls_back_to_default(self) -> None:
        """Unrecognized env var value falls back to the default."""
        assert _flag("drift", default=True) is True
        assert _flag("drift", default=False) is False

    def test_no_env_uses_default(self) -> None:
        """Missing env var uses the provided default."""
        os.environ.pop("TRUSTYAI_ENABLE_NONEXISTENT", None)
        assert _flag("nonexistent", default=True) is True
        assert _flag("nonexistent", default=False) is False
