"""Integration tests for main application endpoint registration."""

import importlib
import os
from http import HTTPStatus
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from trustyai_service.endpoints import routes

# Default app (all flags at default values)
from trustyai_service.main import app, run_server
from trustyai_service.service.config import feature_flags

client = TestClient(app)


def _build_app_with_flags(overrides: dict[str, bool]) -> TestClient:
    """Reload the app module with custom feature flag overrides."""
    with patch.dict(feature_flags.ENDPOINTS, overrides):
        from trustyai_service import main  # noqa: PLC0415

        importlib.reload(main)
        return TestClient(main.app)


class TestAppCoreEndpoints:
    """Test core application endpoints."""

    def test_root_endpoint(self) -> None:
        """Test root endpoint is accessible."""
        response = client.get("/")
        assert response.status_code == HTTPStatus.OK
        assert "message" in response.json()

    def test_health_endpoints(self) -> None:
        """Test health check endpoints are registered."""
        # Readiness probe - may fail if storage not available in test
        response = client.get(routes.HEALTH_READY)
        assert response.status_code in (HTTPStatus.OK, HTTPStatus.SERVICE_UNAVAILABLE)
        payload = response.json()
        assert "checks" in payload
        if response.status_code == HTTPStatus.OK:
            assert payload["status"] == "ready"
        else:
            assert payload["status"] == "not_ready"

        # Liveness probe - should always succeed
        response = client.get(routes.HEALTH_LIVE)
        assert response.status_code == HTTPStatus.OK
        assert response.json()["status"] == "alive"
        assert "checks" in response.json()

    def test_openapi_docs_accessible(self) -> None:
        """Test that OpenAPI documentation is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == HTTPStatus.OK
        openapi = response.json()
        assert "paths" in openapi
        assert "info" in openapi

    def test_prometheus_metrics_endpoint(self) -> None:
        """Test that Prometheus metrics endpoint is accessible."""
        response = client.get(routes.PROMETHEUS_METRICS)
        assert response.status_code == HTTPStatus.OK
        # Prometheus metrics are in text format
        assert "text/plain" in response.headers["content-type"]
        # Check for some standard Prometheus metric format
        content = response.text
        assert len(content) > 0

    def test_trailing_slash_no_redirect(self) -> None:
        """Trailing slash must not 307 redirect (which drops POST bodies)."""
        response = client.post(
            "/metrics/drift/kstest/",
            json={"modelId": "test"},
        )
        assert response.status_code != HTTPStatus.TEMPORARY_REDIRECT
        assert response.status_code != HTTPStatus.NOT_FOUND
        # Should match the same route as without trailing slash
        response_no_slash = client.post(
            routes.DRIFT_KSTEST.compute,
            json={"modelId": "test"},
        )
        assert response.status_code == response_no_slash.status_code

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
        response = client.get(routes.DRIFT_KSTEST.definition)
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
            routes.DRIFT_KSTEST.compute,
            routes.DRIFT_KSTEST.definition,
            routes.DRIFT_KSTEST.request,
            routes.DRIFT_KSTEST.requests,
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
        kstest_compute = openapi["paths"][routes.DRIFT_KSTEST.compute]["post"]
        assert "tags" in kstest_compute
        assert "Drift Metrics: KSTest" in kstest_compute["tags"]

        # Check tags for definition endpoint
        kstest_definition = openapi["paths"][routes.DRIFT_KSTEST.definition]["get"]
        assert "tags" in kstest_definition
        assert "Drift Metrics: KSTest" in kstest_definition["tags"]

        # Check tags for schedule endpoint
        kstest_schedule = openapi["paths"][routes.DRIFT_KSTEST.request]["post"]
        assert "tags" in kstest_schedule
        assert "Drift Metrics: KSTest" in kstest_schedule["tags"]


class TestCompareMeansMetricIntegration:
    """Integration tests for CompareMeans drift metric registration in main app."""

    def test_comparemeans_definition_endpoint_accessible(self) -> None:
        """Test that CompareMeans definition endpoint is accessible."""
        response = client.get(routes.DRIFT_COMPARE_MEANS.definition)
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
            routes.DRIFT_COMPARE_MEANS.compute,
            routes.DRIFT_COMPARE_MEANS.definition,
            routes.DRIFT_COMPARE_MEANS.request,
            routes.DRIFT_COMPARE_MEANS.requests,
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
        comparemeans_compute = openapi["paths"][routes.DRIFT_COMPARE_MEANS.compute][
            "post"
        ]
        assert "tags" in comparemeans_compute
        assert "Drift Metrics: CompareMeans" in comparemeans_compute["tags"]

        # Check tags for definition endpoint
        comparemeans_definition = openapi["paths"][
            routes.DRIFT_COMPARE_MEANS.definition
        ]["get"]
        assert "tags" in comparemeans_definition
        assert "Drift Metrics: CompareMeans" in comparemeans_definition["tags"]

        # Check tags for schedule endpoint
        comparemeans_schedule = openapi["paths"][routes.DRIFT_COMPARE_MEANS.request][
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
            routes.DRIFT_MEANSHIFT.compute,
            routes.DRIFT_MEANSHIFT.definition,
            routes.DRIFT_MEANSHIFT.request,
            routes.DRIFT_MEANSHIFT.requests,
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
        meanshift_compute = openapi["paths"][routes.DRIFT_MEANSHIFT.compute]["post"]
        assert meanshift_compute.get("deprecated") is True

        meanshift_definition = openapi["paths"][routes.DRIFT_MEANSHIFT.definition][
            "get"
        ]
        assert meanshift_definition.get("deprecated") is True

        meanshift_schedule = openapi["paths"][routes.DRIFT_MEANSHIFT.request]["post"]
        assert meanshift_schedule.get("deprecated") is True

        meanshift_delete = openapi["paths"][routes.DRIFT_MEANSHIFT.request]["delete"]
        assert meanshift_delete.get("deprecated") is True

        meanshift_list = openapi["paths"][routes.DRIFT_MEANSHIFT.requests]["get"]
        assert meanshift_list.get("deprecated") is True


class TestFeatureFlagGating:
    """Integration tests for feature flag gating of metric endpoints."""

    def test_all_drift_endpoints_registered_by_default(self) -> None:
        """All drift metric endpoints are registered with default flags."""
        response = client.get("/openapi.json")
        openapi = response.json()
        for route_group in [
            routes.DRIFT_FOURIER_MMD,
            routes.DRIFT_JENSEN_SHANNON,
            routes.DRIFT_KSTEST,
            routes.DRIFT_COMPARE_MEANS,
        ]:
            assert route_group.compute in openapi["paths"], (
                f"{route_group.compute} not found in OpenAPI"
            )

    def test_drift_group_disabled_hides_all_drift(self) -> None:
        """Disabling the drift group flag hides all drift endpoints."""
        test_client = _build_app_with_flags({"drift": False})
        response = test_client.get("/openapi.json")
        openapi = response.json()
        for route_group in [
            routes.DRIFT_FOURIER_MMD,
            routes.DRIFT_JENSEN_SHANNON,
            routes.DRIFT_KSTEST,
            routes.DRIFT_KS_TEST_STREAMING,
            routes.DRIFT_COMPARE_MEANS,
        ]:
            assert route_group.compute not in openapi["paths"], (
                f"{route_group.compute} should be hidden when drift group is disabled"
            )

    def test_individual_drift_metric_disabled(self) -> None:
        """Disabling an individual drift metric hides only that metric."""
        test_client = _build_app_with_flags({"drift_mmd": False})
        response = test_client.get("/openapi.json")
        openapi = response.json()
        assert routes.DRIFT_MMD.compute not in openapi["paths"]
        assert routes.DRIFT_KSTEST.compute in openapi["paths"]
        assert routes.DRIFT_JENSEN_SHANNON.compute in openapi["paths"]

    def test_explainer_disabled_by_default(self) -> None:
        """Explainer endpoints are not registered with default flags."""
        response = client.get("/openapi.json")
        openapi = response.json()
        assert "/explainers/local" not in str(openapi["paths"])

    def test_fairness_disabled_hides_legacy(self) -> None:
        """Disabling fairness also hides legacy /metrics prefixed endpoints."""
        test_client = _build_app_with_flags({"fairness": False})
        response = test_client.get("/openapi.json")
        openapi = response.json()
        legacy_paths = [
            p for p in openapi["paths"] if p.startswith("/metrics/group/fairness")
        ]
        assert legacy_paths == []


class TestPortCollisionGuard:
    """Test that run_server() rejects HEALTH_PORT colliding with HTTP_PORT or SSL_PORT."""

    @pytest.mark.asyncio
    async def test_health_port_equals_http_port(self) -> None:
        """ValueError raised when HEALTH_PORT == HTTP_PORT."""
        with (
            patch.dict(os.environ, {"HEALTH_PORT": "8080", "HTTP_PORT": "8080"}),
            pytest.raises(
                ValueError,
                match=r"HEALTH_PORT \(8080\) must differ from HTTP_PORT \(8080\) and SSL_PORT",
            ),
        ):
            await run_server()

    @pytest.mark.asyncio
    async def test_health_port_equals_ssl_port(self) -> None:
        """ValueError raised when HEALTH_PORT == SSL_PORT."""
        with (
            patch.dict(os.environ, {"HEALTH_PORT": "4443", "SSL_PORT": "4443"}),
            pytest.raises(
                ValueError,
                match=r"HEALTH_PORT \(4443\) must differ from HTTP_PORT .* and SSL_PORT \(4443\)",
            ),
        ):
            await run_server()

    @pytest.mark.asyncio
    async def test_default_ports_no_collision(self) -> None:
        """Non-colliding ports (8081, 4443, 8080) do not trigger the guard — server creation proceeds."""
        with (
            patch.dict(
                os.environ,
                {"HTTP_PORT": "8081", "SSL_PORT": "4443", "HEALTH_PORT": "8080"},
            ),
            patch("trustyai_service.main.PolicyAwareConfig"),
            patch("trustyai_service.main.serve", side_effect=RuntimeError("stop")),
            pytest.raises(RuntimeError, match="stop"),
        ):
            await run_server()

    @pytest.mark.asyncio
    async def test_http_port_equals_ssl_port(self) -> None:
        """ValueError raised when HTTP_PORT == SSL_PORT."""
        with (
            patch.dict(os.environ, {"HTTP_PORT": "8443", "SSL_PORT": "8443"}),
            pytest.raises(
                ValueError,
                match=r"HTTP_PORT \(8443\) must differ from SSL_PORT \(8443\)",
            ),
        ):
            await run_server()

    @pytest.mark.asyncio
    async def test_all_three_ports_equal(self) -> None:
        """ValueError raised when all three ports are equal."""
        with (
            patch.dict(
                os.environ,
                {"HEALTH_PORT": "9000", "HTTP_PORT": "9000", "SSL_PORT": "9000"},
            ),
            pytest.raises(
                ValueError,
                match=r"HEALTH_PORT \(9000\) must differ from HTTP_PORT \(9000\) and SSL_PORT \(9000\)",
            ),
        ):
            await run_server()
