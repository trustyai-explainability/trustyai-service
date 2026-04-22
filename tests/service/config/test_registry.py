"""Unit tests for registry.py."""

from unittest.mock import MagicMock, patch

from src.service.config.registry import (
    register_if_enabled,
    register_if_enabled_with_group,
)


class TestRegisterIfEnabled:
    def test_registers_router_when_flag_enabled(self):
        app = MagicMock()
        router = MagicMock()

        with patch.dict("src.service.config.registry.ENDPOINTS", {"fairness": True, "other": False}, clear=True):
            register_if_enabled(app, router, "fairness", "Test Tag")

        app.include_router.assert_called_once_with(router, tags=["Test Tag"])

    def test_skips_router_when_flag_disabled(self):
        app = MagicMock()
        router = MagicMock()

        with patch.dict("src.service.config.registry.ENDPOINTS", {"fairness": False, "other": True}, clear=True):
            register_if_enabled(app, router, "fairness", "Test Tag")

        app.include_router.assert_not_called()

    def test_registers_with_prefix(self):
        app = MagicMock()
        router = MagicMock()

        with patch.dict("src.service.config.registry.ENDPOINTS", {"fairness": True}, clear=True):
            register_if_enabled(app, router, "fairness", prefix="/metrics")

        app.include_router.assert_called_once_with(router, prefix="/metrics")

    def test_registers_with_tag_and_prefix(self):
        app = MagicMock()
        router = MagicMock()

        with patch.dict("src.service.config.registry.ENDPOINTS", {"fairness": True}, clear=True):
            register_if_enabled(app, router, "fairness", "Test Tag", prefix="/metrics")

        app.include_router.assert_called_once_with(router, tags=["Test Tag"], prefix="/metrics")

    def test_no_tag_no_prefix(self):
        app = MagicMock()
        router = MagicMock()

        with patch.dict("src.service.config.registry.ENDPOINTS", {"drift": True}, clear=True):
            register_if_enabled(app, router, "drift")

        app.include_router.assert_called_once_with(router)


class TestRegisterIfEnabledWithGroup:
    def test_registers_when_both_flags_enabled(self):
        """Both group and metric flag enabled -> registers."""
        app = MagicMock()
        router = MagicMock()

        with patch.dict(
            "src.service.config.registry.ENDPOINTS",
            {"drift": True, "drift_jensen_shannon": True},
            clear=True,
        ):
            register_if_enabled_with_group(
                app,
                router,
                "drift",
                "drift_jensen_shannon",
                "Drift Metrics: JensenShannon",
            )

        app.include_router.assert_called_once()

    def test_skips_when_group_flag_disabled(self):
        """Group disabled -> skips regardless of metric flag."""
        app = MagicMock()
        router = MagicMock()

        with patch.dict(
            "src.service.config.registry.ENDPOINTS",
            {"drift": False, "drift_jensen_shannon": True},
            clear=True,
        ):
            register_if_enabled_with_group(
                app,
                router,
                "drift",
                "drift_jensen_shannon",
                "Drift Metrics: JensenShannon",
            )

        app.include_router.assert_not_called()

    def test_skips_when_metric_flag_disabled(self):
        """Group enabled but metric flag disabled -> skips just that metric."""
        app = MagicMock()
        router = MagicMock()

        with patch.dict(
            "src.service.config.registry.ENDPOINTS",
            {"drift": True, "drift_jensen_shannon": False},
            clear=True,
        ):
            register_if_enabled_with_group(
                app,
                router,
                "drift",
                "drift_jensen_shannon",
                "Drift Metrics: JensenShannon",
            )

        app.include_router.assert_not_called()

    def test_other_metrics_still_register_when_one_is_disabled(self):
        """Disabling one metric doesn't affect other metrics in the same group."""
        app = MagicMock()
        ks_router = MagicMock()
        js_router = MagicMock()

        with patch.dict(
            "src.service.config.registry.ENDPOINTS",
            {"drift": True, "drift_ks_test": True, "drift_jensen_shannon": False},
            clear=True,
        ):
            register_if_enabled_with_group(
                app,
                ks_router,
                "drift",
                "drift_ks_test",
                "Drift Metrics: KSTest",
            )
            register_if_enabled_with_group(
                app,
                js_router,
                "drift",
                "drift_jensen_shannon",
                "Drift Metrics: JensenShannon",
            )

        # KS-Test registered
        app.include_router.assert_any_call(ks_router, tags=["Drift Metrics: KSTest"])
        # Jensen-Shannon skipped (called once for KS-Test only)
        assert app.include_router.call_count == 1

    def test_registers_with_prefix(self):
        app = MagicMock()
        router = MagicMock()

        with patch.dict(
            "src.service.config.registry.ENDPOINTS",
            {"fairness": True, "fairness_spd": True},
            clear=True,
        ):
            register_if_enabled_with_group(
                app,
                router,
                "fairness",
                "fairness_spd",
                prefix="/metrics",
            )

        app.include_router.assert_called_once_with(router, prefix="/metrics")

    def test_registers_with_tag_and_prefix(self):
        app = MagicMock()
        router = MagicMock()

        with patch.dict(
            "src.service.config.registry.ENDPOINTS",
            {"fairness": True, "fairness_dir": True},
            clear=True,
        ):
            register_if_enabled_with_group(
                app,
                router,
                "fairness",
                "fairness_dir",
                "Fairness: DIR",
                prefix="/metrics",
            )

        app.include_router.assert_called_once_with(
            router,
            tags=["Fairness: DIR"],
            prefix="/metrics",
        )

    def test_explainer_group_gates_individual_metrics(self):
        """Group disabled -> explainers skipped regardless of individual flags."""
        app = MagicMock()
        router = MagicMock()

        with patch.dict(
            "src.service.config.registry.ENDPOINTS",
            {"explainer": False, "explainer_local": True},
            clear=True,
        ):
            register_if_enabled_with_group(
                app,
                router,
                "explainer",
                "explainer_local",
                "Explainers: Local",
            )

        app.include_router.assert_not_called()

    def test_explainer_individual_flag_gates_specific_explainer(self):
        """Group enabled but individual flag disabled -> skips just that explainer."""
        app = MagicMock()
        local_router = MagicMock()
        global_router = MagicMock()

        with patch.dict(
            "src.service.config.registry.ENDPOINTS",
            {"explainer": True, "explainer_local": False, "explainer_global": True},
            clear=True,
        ):
            register_if_enabled_with_group(
                app,
                local_router,
                "explainer",
                "explainer_local",
                "Explainers: Local",
            )
            register_if_enabled_with_group(
                app,
                global_router,
                "explainer",
                "explainer_global",
                "Explainers: Global",
            )

        # Global registered
        app.include_router.assert_any_call(global_router, tags=["Explainers: Global"])
        # Local skipped (called once for Global only)
        assert app.include_router.call_count == 1
