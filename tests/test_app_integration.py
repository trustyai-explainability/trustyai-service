"""Integration tests for main application endpoint registration."""

from http import HTTPStatus
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.endpoints.metrics.drift.compare_means import (
    router as drift_comparemeans_router,
)
from src.endpoints.metrics.drift.jensen_shannon import (
    router as drift_jensenshannon_router,
)
from src.endpoints.metrics.drift.kolmogorov_smirnov import (
    router as drift_kstest_router,
)
from src.endpoints.metrics.fairness.group.dir import router as dir_router
from src.endpoints.metrics.fairness.group.spd import router as spd_router
from src.main import app
from src.service.config.feature_flags import ENDPOINTS
from src.service.config.registry import register_if_enabled_with_group

client = TestClient(app)


class TestAppCoreEndpoints:
    """Test core application endpoints."""

    def test_root_endpoint(self) -> None:
        """Test root endpoint is accessible."""
        response = client.get("/")
        assert response.status_code == HTTPStatus.OK
        assert "message" in response.json()

    def test_health_endpoints(self) -> None:
        """Test health check endpoints are registered."""
        # Readiness probe
        response = client.get("/q/health/ready")
        assert response.status_code == HTTPStatus.OK
        assert response.json()["status"] == "ready"

        # Liveness probe
        response = client.get("/q/health/live")
        assert response.status_code == HTTPStatus.OK
        assert response.json()["status"] == "live"

    def test_openapi_docs_accessible(self) -> None:
        """Test that OpenAPI documentation is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        openapi = response.json()
        assert "paths" in openapi
        assert "info" in openapi

    def test_prometheus_metrics_endpoint(self) -> None:
        """Test that Prometheus metrics endpoint is accessible."""
        response = client.get("/q/metrics")
        assert response.status_code == HTTPStatus.OK
        # Prometheus metrics are in text format
        assert "text/plain" in response.headers["content-type"]
        # Check for some standard Prometheus metric format
        content = response.text
        assert len(content) > 0

    def test_cors_headers(self) -> None:
        """Test that CORS headers are properly configured."""
        # CORS headers are added by middleware but may not appear in TestClient
        # unless an origin is specified. Test with an OPTIONS request.
        response = client.options("/", headers={"Origin": "http://example.com"})
        # CORS middleware should allow the request
        assert response.status_code in [
            HTTPStatus.OK,
            HTTPStatus.METHOD_NOT_ALLOWED,
        ]  # OPTIONS may not be defined but CORS should process it


class TestKSTestMetricIntegration:
    """Integration tests for KSTest drift metric registration in main app."""

    def test_kstest_definition_endpoint_accessible(self) -> None:
        """Test that KSTest definition endpoint is accessible."""
        response = client.get("/metrics/drift/kstest/definition")
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert "name" in data
        assert "description" in data
        assert "Kolmogorov-Smirnov" in data["name"]

    def test_kstest_endpoints_in_openapi(self) -> None:
        """Test that all KSTest endpoints are documented in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        openapi = response.json()

        # Check that all KSTest endpoints are documented
        expected_paths = [
            "/metrics/drift/kstest",
            "/metrics/drift/kstest/definition",
            "/metrics/drift/kstest/request",
            "/metrics/drift/kstest/requests",
        ]

        for path in expected_paths:
            assert path in openapi["paths"], (
                f"Expected path {path} not found in OpenAPI documentation"
            )

    def test_kstest_openapi_tags(self) -> None:
        """Test that KSTest endpoints have correct tags in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        openapi = response.json()

        # Check tags for compute endpoint
        kstest_compute = openapi["paths"]["/metrics/drift/kstest"]["post"]
        assert "tags" in kstest_compute
        assert "Drift Metrics: KSTest" in kstest_compute["tags"]

        # Check tags for definition endpoint
        kstest_definition = openapi["paths"]["/metrics/drift/kstest/definition"]["get"]
        assert "tags" in kstest_definition
        assert "Drift Metrics: KSTest" in kstest_definition["tags"]

        # Check tags for schedule endpoint
        kstest_schedule = openapi["paths"]["/metrics/drift/kstest/request"]["post"]
        assert "tags" in kstest_schedule
        assert "Drift Metrics: KSTest" in kstest_schedule["tags"]


class TestCompareMeansMetricIntegration:
    """Integration tests for CompareMeans drift metric registration in main app."""

    def test_comparemeans_definition_endpoint_accessible(self) -> None:
        """Test that CompareMeans definition endpoint is accessible."""
        response = client.get("/metrics/drift/comparemeans/definition")
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert "name" in data
        assert "description" in data
        assert "T-Test" in data["name"]

    def test_comparemeans_endpoints_in_openapi(self) -> None:
        """Test that all CompareMeans endpoints are documented in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        openapi = response.json()

        # Check that all CompareMeans endpoints are documented
        expected_paths = [
            "/metrics/drift/comparemeans",
            "/metrics/drift/comparemeans/definition",
            "/metrics/drift/comparemeans/request",
            "/metrics/drift/comparemeans/requests",
        ]

        for path in expected_paths:
            assert path in openapi["paths"], (
                f"Expected path {path} not found in OpenAPI documentation"
            )

    def test_comparemeans_openapi_tags(self) -> None:
        """Test that CompareMeans endpoints have correct tags in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        openapi = response.json()

        # Check tags for compute endpoint
        comparemeans_compute = openapi["paths"]["/metrics/drift/comparemeans"]["post"]
        assert "tags" in comparemeans_compute
        assert "Drift Metrics: CompareMeans" in comparemeans_compute["tags"]

        # Check tags for definition endpoint
        comparemeans_definition = openapi["paths"][
            "/metrics/drift/comparemeans/definition"
        ]["get"]
        assert "tags" in comparemeans_definition
        assert "Drift Metrics: CompareMeans" in comparemeans_definition["tags"]

        # Check tags for schedule endpoint
        comparemeans_schedule = openapi["paths"]["/metrics/drift/comparemeans/request"][
            "post"
        ]
        assert "tags" in comparemeans_schedule
        assert "Drift Metrics: CompareMeans" in comparemeans_schedule["tags"]

    def test_deprecated_meanshift_endpoints_in_openapi(self) -> None:
        """Test that deprecated Meanshift endpoints are documented in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        openapi = response.json()

        # Check that all deprecated Meanshift endpoints are documented
        expected_paths = [
            "/metrics/drift/meanshift",
            "/metrics/drift/meanshift/definition",
            "/metrics/drift/meanshift/request",
            "/metrics/drift/meanshift/requests",
        ]

        for path in expected_paths:
            assert path in openapi["paths"], (
                f"Expected deprecated path {path} not found in OpenAPI documentation"
            )

    def test_deprecated_meanshift_endpoints_marked_deprecated(self) -> None:
        """Meanshift endpoints are marked as `deprecated` in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        openapi = response.json()

        # Check that Meanshift endpoints are marked as deprecated
        meanshift_compute = openapi["paths"]["/metrics/drift/meanshift"]["post"]
        assert meanshift_compute.get("deprecated") is True

        meanshift_definition = openapi["paths"]["/metrics/drift/meanshift/definition"][
            "get"
        ]
        assert meanshift_definition.get("deprecated") is True

        meanshift_schedule = openapi["paths"]["/metrics/drift/meanshift/request"][
            "post"
        ]
        assert meanshift_schedule.get("deprecated") is True

        meanshift_delete = openapi["paths"]["/metrics/drift/meanshift/request"][
            "delete"
        ]
        assert meanshift_delete.get("deprecated") is True

        meanshift_list = openapi["paths"]["/metrics/drift/meanshift/requests"]["get"]
        assert meanshift_list.get("deprecated") is True


def _build_app_with_flags(overrides: dict[str, bool]) -> FastAPI:
    """Build a minimal FastAPI app with custom feature flag overrides.

    Constructs a fresh app and registers drift/fairness routers using
    the same ``register_if_enabled_with_group`` helper that ``src/main.py``
    uses, but with patched flags.
    """
    test_app = FastAPI()
    patched = {**ENDPOINTS, **overrides}

    with patch.dict("src.service.config.registry.ENDPOINTS", patched, clear=True):
        register_if_enabled_with_group(
            test_app,
            dir_router,
            "fairness",
            "fairness_dir",
            "Fairness: DIR",
        )
        register_if_enabled_with_group(
            test_app,
            spd_router,
            "fairness",
            "fairness_spd",
            "Fairness: SPD",
        )
        register_if_enabled_with_group(
            test_app,
            drift_comparemeans_router,
            "drift",
            "drift_compare_means",
            "Drift: CompareMeans",
        )
        register_if_enabled_with_group(
            test_app,
            drift_kstest_router,
            "drift",
            "drift_ks_test",
            "Drift: KSTest",
        )
        register_if_enabled_with_group(
            test_app,
            drift_jensenshannon_router,
            "drift",
            "drift_jensen_shannon",
            "Drift: JensenShannon",
        )
        register_if_enabled_with_group(
            test_app,
            dir_router,
            "fairness",
            "fairness_dir",
            prefix="/metrics",
            tag="{Legacy}: DIR",
        )
        register_if_enabled_with_group(
            test_app,
            spd_router,
            "fairness",
            "fairness_spd",
            prefix="/metrics",
            tag="{Legacy}: SPD",
        )

    return test_app


class TestFeatureFlagGating:
    """Verify that disabling a feature flag removes endpoints from OpenAPI."""

    def test_disabling_fairness_removes_dir_and_spd(self) -> None:
        """Setting fairness=False removes all fairness endpoints."""
        test_app = _build_app_with_flags({"fairness": False})
        test_client = TestClient(test_app)

        response = test_client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        paths = response.json()["paths"]

        fairness_paths = [
            p for p in paths if "/fairness/" in p or "/dir" in p or "/spd" in p
        ]
        assert fairness_paths == []

    def test_disabling_drift_removes_all_drift_endpoints(self) -> None:
        """Setting drift=False removes KSTest, CompareMeans, and JensenShannon."""
        test_app = _build_app_with_flags({"drift": False})
        test_client = TestClient(test_app)

        response = test_client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        paths = response.json()["paths"]

        drift_paths = [p for p in paths if "/metrics/drift/" in p]
        assert drift_paths == []

    def test_disabling_individual_metric_keeps_others(self) -> None:
        """Disabling drift_ks_test removes KS but keeps CompareMeans."""
        test_app = _build_app_with_flags({"drift_ks_test": False})
        test_client = TestClient(test_app)

        response = test_client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        paths = response.json()["paths"]

        assert "/metrics/drift/kstest" not in paths
        assert "/metrics/drift/comparemeans" in paths
