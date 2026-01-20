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
