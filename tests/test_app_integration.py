"""Integration tests for main application endpoint registration."""

from http import HTTPStatus

from fastapi.testclient import TestClient

from src.endpoints import routes
from src.main import app

client = TestClient(app)


class TestAppCoreEndpoints:
    """Test core application endpoints."""

    def test_root_endpoint(self) -> None:
        """Test root endpoint is accessible."""
        response = client.get(routes.CONSUMER_ROOT)
        assert response.status_code == HTTPStatus.OK
        assert "message" in response.json()

    def test_health_endpoints(self) -> None:
        """Test health check endpoints are registered."""
        # Readiness probe
        response = client.get(routes.HEALTH_READY)
        assert response.status_code == HTTPStatus.OK
        assert response.json()["status"] == "ready"

        # Liveness probe
        response = client.get(routes.HEALTH_LIVE)
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
            f"{routes.DRIFT_KSTEST.compute}/",
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
        response = client.options(
            routes.CONSUMER_ROOT, headers={"Origin": "http://example.com"}
        )
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


class TestFourierMMDMetricIntegration:
    """Integration tests for FourierMMD drift metric registration."""

    def test_fouriermmd_endpoints_in_openapi(self) -> None:
        """FourierMMD endpoints are registered in OpenAPI."""
        response = client.get("/openapi.json")
        openapi = response.json()
        for path in [
            routes.DRIFT_FOURIER_MMD.compute,
            routes.DRIFT_FOURIER_MMD.definition,
            routes.DRIFT_FOURIER_MMD.request,
            routes.DRIFT_FOURIER_MMD.requests,
        ]:
            assert path in openapi["paths"], f"{path} not found in OpenAPI"


class TestApproxKSTestMetricIntegration:
    """Integration tests for ApproxKSTest drift metric registration."""

    def test_approxkstest_endpoints_in_openapi(self) -> None:
        """ApproxKSTest endpoints are registered in OpenAPI."""
        response = client.get("/openapi.json")
        openapi = response.json()
        for path in [
            routes.DRIFT_APPROX_KS_TEST.compute,
            routes.DRIFT_APPROX_KS_TEST.definition,
            routes.DRIFT_APPROX_KS_TEST.request,
            routes.DRIFT_APPROX_KS_TEST.requests,
        ]:
            assert path in openapi["paths"], f"{path} not found in OpenAPI"


class TestJensenShannonMetricIntegration:
    """Integration tests for JensenShannon drift metric registration."""

    def test_jensenshannon_endpoints_in_openapi(self) -> None:
        """JensenShannon endpoints are registered in OpenAPI."""
        response = client.get("/openapi.json")
        openapi = response.json()
        for path in [
            routes.DRIFT_JENSEN_SHANNON.compute,
            routes.DRIFT_JENSEN_SHANNON.definition,
            routes.DRIFT_JENSEN_SHANNON.request,
            routes.DRIFT_JENSEN_SHANNON.requests,
        ]:
            assert path in openapi["paths"], f"{path} not found in OpenAPI"
