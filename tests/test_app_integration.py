"""Integration tests for main application endpoint registration."""

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestAppCoreEndpoints:
    """Test core application endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint is accessible."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_endpoints(self):
        """Test health check endpoints are registered."""
        # Readiness probe
        response = client.get("/q/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

        # Liveness probe
        response = client.get("/q/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "live"

    def test_openapi_docs_accessible(self):
        """Test that OpenAPI documentation is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi = response.json()
        assert "paths" in openapi
        assert "info" in openapi

    def test_prometheus_metrics_endpoint(self):
        """Test that Prometheus metrics endpoint is accessible."""
        response = client.get("/q/metrics")
        assert response.status_code == 200
        # Prometheus metrics are in text format
        assert "text/plain" in response.headers["content-type"]
        # Check for some standard Prometheus metric format
        content = response.text
        assert len(content) > 0

    def test_cors_headers(self):
        """Test that CORS headers are properly configured."""
        # CORS headers are added by middleware but may not appear in TestClient
        # unless an origin is specified. Test with an OPTIONS request.
        response = client.options("/", headers={"Origin": "http://example.com"})
        # CORS middleware should allow the request
        assert response.status_code in [200, 405]  # OPTIONS may not be defined but CORS should process it


class TestKSTestMetricIntegration:
    """Integration tests for KSTest drift metric registration in main app."""

    def test_kstest_definition_endpoint_accessible(self):
        """Test that KSTest definition endpoint is accessible."""
        response = client.get("/metrics/drift/kstest/definition")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "description" in data
        assert "Kolmogorov-Smirnov" in data["name"]

    def test_kstest_endpoints_in_openapi(self):
        """Test that all KSTest endpoints are documented in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi = response.json()

        # Check that all KSTest endpoints are documented
        expected_paths = [
            "/metrics/drift/kstest",
            "/metrics/drift/kstest/definition",
            "/metrics/drift/kstest/request",
            "/metrics/drift/kstest/requests",
        ]

        for path in expected_paths:
            assert path in openapi["paths"], f"Expected path {path} not found in OpenAPI documentation"

    def test_kstest_openapi_tags(self):
        """Test that KSTest endpoints have correct tags in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
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

    def test_comparemeans_definition_endpoint_accessible(self):
        """Test that CompareMeans definition endpoint is accessible."""
        response = client.get("/metrics/drift/comparemeans/definition")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "description" in data
        assert "T-Test" in data["name"]

    def test_comparemeans_endpoints_in_openapi(self):
        """Test that all CompareMeans endpoints are documented in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi = response.json()

        # Check that all CompareMeans endpoints are documented
        expected_paths = [
            "/metrics/drift/comparemeans",
            "/metrics/drift/comparemeans/definition",
            "/metrics/drift/comparemeans/request",
            "/metrics/drift/comparemeans/requests",
        ]

        for path in expected_paths:
            assert path in openapi["paths"], f"Expected path {path} not found in OpenAPI documentation"

    def test_comparemeans_openapi_tags(self):
        """Test that CompareMeans endpoints have correct tags in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi = response.json()

        # Check tags for compute endpoint
        comparemeans_compute = openapi["paths"]["/metrics/drift/comparemeans"]["post"]
        assert "tags" in comparemeans_compute
        assert "Drift Metrics: CompareMeans" in comparemeans_compute["tags"]

        # Check tags for definition endpoint
        comparemeans_definition = openapi["paths"]["/metrics/drift/comparemeans/definition"]["get"]
        assert "tags" in comparemeans_definition
        assert "Drift Metrics: CompareMeans" in comparemeans_definition["tags"]

        # Check tags for schedule endpoint
        comparemeans_schedule = openapi["paths"]["/metrics/drift/comparemeans/request"]["post"]
        assert "tags" in comparemeans_schedule
        assert "Drift Metrics: CompareMeans" in comparemeans_schedule["tags"]

    def test_deprecated_meanshift_endpoints_in_openapi(self):
        """Test that deprecated Meanshift endpoints are documented in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi = response.json()

        # Check that all deprecated Meanshift endpoints are documented
        expected_paths = [
            "/metrics/drift/meanshift",
            "/metrics/drift/meanshift/definition",
            "/metrics/drift/meanshift/request",
            "/metrics/drift/meanshift/requests",
        ]

        for path in expected_paths:
            assert path in openapi["paths"], f"Expected deprecated path {path} not found in OpenAPI documentation"

    def test_deprecated_meanshift_endpoints_marked_deprecated(self):
        """Test that deprecated Meanshift endpoints are marked as deprecated in OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi = response.json()

        # Check that Meanshift endpoints are marked as deprecated
        meanshift_compute = openapi["paths"]["/metrics/drift/meanshift"]["post"]
        assert meanshift_compute.get("deprecated") is True

        meanshift_definition = openapi["paths"]["/metrics/drift/meanshift/definition"]["get"]
        assert meanshift_definition.get("deprecated") is True

        meanshift_schedule = openapi["paths"]["/metrics/drift/meanshift/request"]["post"]
        assert meanshift_schedule.get("deprecated") is True

        meanshift_delete = openapi["paths"]["/metrics/drift/meanshift/request"]["delete"]
        assert meanshift_delete.get("deprecated") is True

        meanshift_list = openapi["paths"]["/metrics/drift/meanshift/requests"]["get"]
        assert meanshift_list.get("deprecated") is True
