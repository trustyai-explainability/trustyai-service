"""Tests for router registration helpers."""

from unittest.mock import MagicMock, patch

from trustyai_service.service.config import registry
from trustyai_service.service.config.registry import (
    register_if_enabled,
    register_if_enabled_with_group,
)


def _make_app_and_router() -> tuple[MagicMock, MagicMock]:
    return MagicMock(), MagicMock()


class TestRegisterIfEnabled:
    """Tests for single-flag endpoint registration."""

    def test_enabled_flag_registers_router(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(registry.feature_flags.ENDPOINTS, {"drift": True}):
            register_if_enabled(app, router, "drift", tag="Drift")
        app.include_router.assert_called_once_with(router, tags=["Drift"])

    def test_disabled_flag_skips_router(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(registry.feature_flags.ENDPOINTS, {"drift": False}):
            register_if_enabled(app, router, "drift", tag="Drift")
        app.include_router.assert_not_called()

    def test_unknown_flag_skips_router(self) -> None:
        app, router = _make_app_and_router()
        register_if_enabled(app, router, "nonexistent", tag="X")
        app.include_router.assert_not_called()

    def test_prefix_passed_to_include_router(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(registry.feature_flags.ENDPOINTS, {"drift": True}):
            register_if_enabled(app, router, "drift", prefix="/metrics")
        app.include_router.assert_called_once_with(router, prefix="/metrics")

    def test_tag_and_prefix_passed_together(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(registry.feature_flags.ENDPOINTS, {"drift": True}):
            register_if_enabled(app, router, "drift", tag="Drift", prefix="/metrics")
        app.include_router.assert_called_once_with(
            router, tags=["Drift"], prefix="/metrics"
        )


class TestRegisterIfEnabledWithGroup:
    """Tests for two-level (group + metric) endpoint registration."""

    def test_both_flags_enabled_registers(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(
            registry.feature_flags.ENDPOINTS,
            {"drift": True, "drift_ks_test": True},
        ):
            register_if_enabled_with_group(
                app, router, "drift", "drift_ks_test", tag="KSTest"
            )
        app.include_router.assert_called_once()

    def test_group_disabled_skips(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(
            registry.feature_flags.ENDPOINTS,
            {"drift": False, "drift_ks_test": True},
        ):
            register_if_enabled_with_group(
                app, router, "drift", "drift_ks_test", tag="KSTest"
            )
        app.include_router.assert_not_called()

    def test_metric_disabled_skips(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(
            registry.feature_flags.ENDPOINTS,
            {"drift": True, "drift_ks_test": False},
        ):
            register_if_enabled_with_group(
                app, router, "drift", "drift_ks_test", tag="KSTest"
            )
        app.include_router.assert_not_called()

    def test_both_disabled_skips(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(
            registry.feature_flags.ENDPOINTS,
            {"drift": False, "drift_ks_test": False},
        ):
            register_if_enabled_with_group(
                app, router, "drift", "drift_ks_test", tag="KSTest"
            )
        app.include_router.assert_not_called()

    def test_explainer_group_gating(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(
            registry.feature_flags.ENDPOINTS,
            {"explainer": False, "explainer_local": True},
        ):
            register_if_enabled_with_group(
                app, router, "explainer", "explainer_local", tag="Local"
            )
        app.include_router.assert_not_called()

    def test_explainer_enabled_registers(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(
            registry.feature_flags.ENDPOINTS,
            {"explainer": True, "explainer_local": True},
        ):
            register_if_enabled_with_group(
                app, router, "explainer", "explainer_local", tag="Local"
            )
        app.include_router.assert_called_once()

    def test_prefix_with_group(self) -> None:
        app, router = _make_app_and_router()
        with patch.dict(
            registry.feature_flags.ENDPOINTS,
            {"fairness": True, "fairness_dir": True},
        ):
            register_if_enabled_with_group(
                app,
                router,
                "fairness",
                "fairness_dir",
                tag="Legacy DIR",
                prefix="/metrics",
            )
        app.include_router.assert_called_once_with(
            router, tags=["Legacy DIR"], prefix="/metrics"
        )
