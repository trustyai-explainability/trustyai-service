"""Tests for health check endpoints and logic."""

import os
import sys
import time
from collections.abc import Generator
from http import HTTPStatus
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from trustyai_service.main import app, health_app
from trustyai_service.service.health_checks import (
    STATUS_ERROR,
    STATUS_OK,
    HealthCache,
    HealthCheck,
    _health_cache,
    check_application_liveness,
    check_http_server,
    check_migration_readiness,
    check_storage_readiness,
    perform_liveness_checks,
    perform_readiness_checks,
)

_MARIA_ENV = {
    "SERVICE_STORAGE_FORMAT": "MARIA",
    "DATABASE_HOST": "localhost",
    "DATABASE_PORT": "3306",
    "DATABASE_USERNAME": "test_user",
    "DATABASE_PASSWORD": "test_pass",  # pragma: allowlist secret
    "DATABASE_DATABASE": "test_db",
}


@pytest.fixture(autouse=True)
def _fake_mariadb_module() -> Generator[None, None, None]:
    """Provide a temporary mariadb module for patching in tests.

    Creates a fake mariadb module when the mariadb extra is not installed,
    allowing @patch("mariadb.connect") to work. Restores original state
    after the test to prevent contamination across the test session.
    """
    original = sys.modules.get("mariadb")
    if original is None:
        fake_mariadb = ModuleType("mariadb")
        fake_mariadb.Error = type("Error", (Exception,), {})  # type: ignore[attr-defined]
        fake_mariadb.connect = MagicMock()  # type: ignore[attr-defined]
        sys.modules["mariadb"] = fake_mariadb
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("mariadb", None)
        else:
            sys.modules["mariadb"] = original


@pytest.fixture(autouse=True)
def _clear_health_cache() -> Generator[None, None, None]:
    """Clear global health cache before and after each test to prevent interference."""
    _health_cache.cache.clear()
    yield
    _health_cache.cache.clear()


@pytest.fixture
def mock_maria_conn() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Mock MariaConnectionManager for health check tests."""
    with (
        patch(
            "trustyai_service.service.data.storage.maria.utils.MariaConnectionManager.__init__",
            return_value=None,
        ) as mock_init,
        patch(
            "trustyai_service.service.data.storage.maria.utils.MariaConnectionManager.__exit__",
            return_value=False,
        ),
        patch(
            "trustyai_service.service.data.storage.maria.utils.MariaConnectionManager.__enter__",
        ) as mock_enter,
    ):
        yield mock_init, mock_enter


class TestHealthCache:
    """Test HealthCache TTL caching."""

    def test_cache_stores_value(self) -> None:
        """Test cache stores and returns values."""
        cache = HealthCache(ttl_seconds=10)
        call_count = 0

        def compute() -> str:
            nonlocal call_count
            call_count += 1
            return "computed_value"

        # First call should compute
        result1 = cache.get_or_compute("key1", compute)
        assert result1 == "computed_value"
        assert call_count == 1

        # Second call should use cache
        result2 = cache.get_or_compute("key1", compute)
        assert result2 == "computed_value"
        assert call_count == 1  # Not incremented - cache hit

    def test_cache_expires_after_ttl(self) -> None:
        """Test cache expires after TTL."""
        cache = HealthCache(ttl_seconds=0.1)  # 100ms TTL
        call_count = 0

        def compute() -> str:
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        # First call
        result1 = cache.get_or_compute("key1", compute)
        assert result1 == "value_1"
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(0.15)

        # Second call should recompute
        result2 = cache.get_or_compute("key1", compute)
        assert result2 == "value_2"
        assert call_count == 2

    def test_cache_different_keys(self) -> None:
        """Test cache handles different keys independently."""
        cache = HealthCache(ttl_seconds=10)

        result1 = cache.get_or_compute("key1", lambda: "value1")
        result2 = cache.get_or_compute("key2", lambda: "value2")

        assert result1 == "value1"
        assert result2 == "value2"

    def test_cache_statistics(self) -> None:
        """Test cache tracks hits and misses."""
        cache = HealthCache(ttl_seconds=10)

        # First call - miss
        cache.get_or_compute("key1", lambda: "value1")
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

        # Second call - hit
        cache.get_or_compute("key1", lambda: "value1")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

        # Third call - hit
        cache.get_or_compute("key1", lambda: "value1")
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1

        # Different key - miss
        cache.get_or_compute("key2", lambda: "value2")
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2


class TestHealthCheck:
    """Test HealthCheck data class."""

    def test_health_check_creation(self) -> None:
        """Test HealthCheck initialization."""
        check = HealthCheck("Test check", "ok")
        assert check.name == "Test check"
        assert check.status == STATUS_OK
        assert check.data == {}

    def test_health_check_with_data(self) -> None:
        """Test HealthCheck with additional data."""
        check = HealthCheck(
            "Test check", STATUS_ERROR, data={"error": "Something went wrong"}
        )
        assert check.name == "Test check"
        assert check.status == STATUS_ERROR
        assert check.data == {"error": "Something went wrong"}

    def test_health_check_to_dict(self) -> None:
        """Test HealthCheck serialization to dictionary."""
        check = HealthCheck("Test check", "ok")
        result = check.to_dict()
        assert result == {"name": "Test check", "status": "ok"}

    def test_health_check_to_dict_with_data(self) -> None:
        """Test HealthCheck serialization with data."""
        check = HealthCheck(
            "Test check", STATUS_ERROR, data={"error": "Something went wrong"}
        )
        result = check.to_dict()
        assert result == {
            "name": "Test check",
            "status": STATUS_ERROR,
            "data": {"error": "Something went wrong"},
        }


class TestStorageAndHealthChecks:
    """Test health check functions."""

    def test_check_http_server(self) -> None:
        """Test HTTP server health check always returns UP."""
        check = check_http_server()
        assert check.name == "HTTP server"
        assert check.status == STATUS_OK

    def test_check_application_liveness(self) -> None:
        """Test application liveness check always returns UP."""
        check = check_application_liveness()
        assert check.name == "Application"
        assert check.status == STATUS_OK

    def test_check_pvc_storage_success(self, tmp_path) -> None:
        """Test PVC storage check succeeds when path exists and is writable."""
        with patch.dict(
            os.environ,
            {"SERVICE_STORAGE_FORMAT": "PVC", "STORAGE_DATA_FOLDER": str(tmp_path)},
        ):
            check = check_storage_readiness()
            assert check.status == STATUS_OK
            assert check.name == "Storage readiness"

    def test_check_pvc_storage_missing_path(self) -> None:
        """Test PVC storage check fails when path doesn't exist."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "PVC",
                "STORAGE_DATA_FOLDER": "/nonexistent/path",
            },
        ):
            check = check_storage_readiness()
            assert check.status == STATUS_ERROR
            assert check.name == "Storage readiness"
            assert "not found" in check.data["error"]

    def test_check_pvc_storage_not_writable(self, tmp_path) -> None:
        """Test PVC storage check fails when path is not writable."""
        with (
            patch.dict(
                os.environ,
                {
                    "SERVICE_STORAGE_FORMAT": "PVC",
                    "STORAGE_DATA_FOLDER": str(tmp_path),
                },
            ),
            patch("os.access", return_value=False),
        ):
            check = check_storage_readiness()
            assert check.status == STATUS_ERROR
            assert check.name == "Storage readiness"
            assert "not writable" in check.data["error"]

    @patch("trustyai_service.service.health_checks.MARIADB_AVAILABLE", False)
    def test_check_maria_storage_library_not_installed(self) -> None:
        """Test MariaDB check fails gracefully when library not installed."""
        with patch.dict(os.environ, {"SERVICE_STORAGE_FORMAT": "MARIA"}):
            check = check_storage_readiness()
            assert check.status == STATUS_ERROR
            assert check.name == "Storage readiness"
            assert "not installed" in check.data["error"]

    @patch("trustyai_service.service.health_checks.MARIADB_AVAILABLE", True)
    def test_check_maria_storage_connection_success(self, mock_maria_conn) -> None:
        """Test MariaDB check succeeds when connection works."""
        mock_init, mock_enter = mock_maria_conn
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_enter.return_value = (MagicMock(), mock_cursor)

        with patch.dict(os.environ, _MARIA_ENV):
            check = check_storage_readiness()
            assert check.status == STATUS_OK
            assert check.name == "Storage readiness"

        mock_init.assert_called_once_with(
            user="test_user",
            password="test_pass",  # pragma: allowlist secret
            host="localhost",
            port=3306,
            database="test_db",
            ssl_ca=None,
            connect_timeout=2,
        )

    @patch("trustyai_service.service.health_checks.MARIADB_AVAILABLE", True)
    def test_check_maria_storage_connection_failure(self, mock_maria_conn) -> None:
        """Test MariaDB check fails when connection fails."""
        _mock_init, mock_enter = mock_maria_conn
        mock_enter.side_effect = Exception("Connection refused")

        with patch.dict(os.environ, _MARIA_ENV):
            check = check_storage_readiness()
            assert check.status == STATUS_ERROR
            assert check.name == "Storage readiness"
            assert "Connection refused" in check.data["error"]

    @patch("trustyai_service.service.health_checks.MARIADB_AVAILABLE", True)
    def test_check_maria_storage_network_error(self, mock_maria_conn) -> None:
        """Test MariaDB check handles network errors specifically."""
        _mock_init, mock_enter = mock_maria_conn
        mock_enter.side_effect = OSError("Network unreachable")

        with patch.dict(os.environ, _MARIA_ENV):
            check = check_storage_readiness()
            assert check.status == STATUS_ERROR
            assert check.name == "Storage readiness"
            assert "Network unreachable" in check.data["error"]

    @patch("trustyai_service.service.health_checks.MARIADB_AVAILABLE", True)
    def test_check_maria_storage_database_alias(self, mock_maria_conn) -> None:
        """Test MariaDB check works with SERVICE_STORAGE_FORMAT=DATABASE alias."""
        _mock_init, mock_enter = mock_maria_conn
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_enter.return_value = (MagicMock(), mock_cursor)

        with patch.dict(
            os.environ, {**_MARIA_ENV, "SERVICE_STORAGE_FORMAT": "DATABASE"}
        ):
            check = check_storage_readiness()
            assert check.status == STATUS_OK

    def test_check_storage_unknown_format(self) -> None:
        """Test storage check fails with unknown storage format."""
        with patch.dict(os.environ, {"SERVICE_STORAGE_FORMAT": "UNKNOWN"}):
            check = check_storage_readiness()
            assert check.status == STATUS_ERROR
            assert check.name == "Storage readiness"
            assert "Unknown storage format" in check.data["error"]

    def test_check_migration_not_configured(self) -> None:
        """Test migration check returns OK when not configured."""
        with patch.dict(os.environ, {"SERVICE_STORAGE_FORMAT": "PVC"}):
            check = check_migration_readiness()
            assert check.status == STATUS_OK
            assert check.name == "Migration"

    def test_check_migration_complete(self, mock_maria_conn) -> None:
        """Test migration check returns OK when migration is COMPLETE."""
        _mock_init, mock_enter = mock_maria_conn
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("COMPLETE",)
        mock_enter.return_value = (MagicMock(), mock_cursor)

        with patch.dict(
            os.environ,
            {**_MARIA_ENV, "DATABASE_ATTEMPT_MIGRATION": "true"},
        ):
            check = check_migration_readiness()
            assert check.status == STATUS_OK
            assert check.name == "Migration"

    def test_check_migration_in_progress(self, mock_maria_conn) -> None:
        """Test migration check returns ERROR when migration is IN_PROGRESS."""
        _mock_init, mock_enter = mock_maria_conn
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("IN_PROGRESS",)
        mock_enter.return_value = (MagicMock(), mock_cursor)

        with patch.dict(
            os.environ,
            {**_MARIA_ENV, "DATABASE_ATTEMPT_MIGRATION": "1"},
        ):
            check = check_migration_readiness()
            assert check.status == STATUS_ERROR
            assert "in progress" in check.data["error"]

    def test_check_pvc_storage_production_mode(self) -> None:
        """Test PVC storage check redacts paths in production mode."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "PVC",
                "STORAGE_DATA_FOLDER": "/nonexistent/path",
                "ENVIRONMENT": "production",
            },
        ):
            check = check_storage_readiness()
            assert check.status == STATUS_ERROR
            assert check.name == "Storage readiness"
            assert "/nonexistent/path" not in check.data["error"]
            assert "not accessible" in check.data["error"]


class TestHealthCheckFunctions:
    """Test health check orchestration functions."""

    def test_perform_readiness_checks_all_up(self, tmp_path) -> None:
        """Test perform_readiness_checks returns correct structure."""
        with patch.dict(
            os.environ,
            {"SERVICE_STORAGE_FORMAT": "PVC", "STORAGE_DATA_FOLDER": str(tmp_path)},
            clear=True,
        ):
            _health_cache.cache.clear()  # Clear cache to pick up new env vars
            status, checks = perform_readiness_checks()
            assert status == STATUS_OK
            assert len(checks) == 3
            # Verify all checks have required fields
            assert all("name" in check and "status" in check for check in checks)

    def test_perform_readiness_checks_storage_down(self) -> None:
        """Test perform_readiness_checks when storage check fails."""
        with patch.dict(
            os.environ,
            {"SERVICE_STORAGE_FORMAT": "PVC", "STORAGE_DATA_FOLDER": "/nonexistent"},
        ):
            status, checks = perform_readiness_checks()
            assert status == STATUS_ERROR
            assert len(checks) == 3
            # Storage check should be DOWN
            storage_check = next(c for c in checks if c["name"] == "Storage readiness")
            assert storage_check["status"] == STATUS_ERROR
            # HTTP server check should be UP
            http_check = next(c for c in checks if c["name"] == "HTTP server")
            assert http_check["status"] == STATUS_OK

    def test_perform_liveness_checks(self) -> None:
        """Test perform_liveness_checks always returns UP."""
        status, checks = perform_liveness_checks()
        assert status == STATUS_OK
        assert len(checks) == 1
        assert checks[0]["name"] == "Application"
        assert checks[0]["status"] == STATUS_OK


class TestHealthEndpoints:
    """Test FastAPI health endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_readiness_endpoint_success(self, client, tmp_path) -> None:
        """Test /q/health/ready endpoint returns success when ready."""
        with patch.dict(
            os.environ,
            {"SERVICE_STORAGE_FORMAT": "PVC", "STORAGE_DATA_FOLDER": str(tmp_path)},
            clear=True,
        ):
            _health_cache.cache.clear()  # Clear cache to pick up new env vars
            response = client.get("/q/health/ready")
            data = response.json()
            assert response.status_code == HTTPStatus.OK
            assert data["status"] == "ready"
            assert len(data["checks"]) == 3
            # Verify all checks passed
            assert all(check["status"] == STATUS_OK for check in data["checks"])

    def test_readiness_endpoint_failure(self, client) -> None:
        """Test /q/health/ready endpoint when not ready."""
        with patch.dict(
            os.environ,
            {"SERVICE_STORAGE_FORMAT": "PVC", "STORAGE_DATA_FOLDER": "/nonexistent"},
        ):
            response = client.get("/q/health/ready")
            assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
            data = response.json()
            assert data["status"] == "not_ready"
            assert len(data["checks"]) == 3

    def test_liveness_endpoint(self, client) -> None:
        """Test /q/health/live endpoint."""
        response = client.get("/q/health/live")
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert data["status"] == "alive"
        assert len(data["checks"]) == 1
        assert data["checks"][0]["name"] == "Application"

    def test_general_health_endpoint_success(self, client, tmp_path) -> None:
        """Test /q/health endpoint returns healthy when all checks pass."""
        with patch.dict(
            os.environ,
            {"SERVICE_STORAGE_FORMAT": "PVC", "STORAGE_DATA_FOLDER": str(tmp_path)},
            clear=True,
        ):
            _health_cache.cache.clear()  # Clear cache to pick up new env vars
            response = client.get("/q/health")
            data = response.json()
            assert response.status_code == HTTPStatus.OK
            assert data["status"] == "healthy"
            # Should have both readiness and liveness checks
            assert "readiness" in data["checks"]
            assert "liveness" in data["checks"]

    def test_general_health_endpoint_failure(self, client) -> None:
        """Test /q/health endpoint when readiness fails."""
        with patch.dict(
            os.environ,
            {"SERVICE_STORAGE_FORMAT": "PVC", "STORAGE_DATA_FOLDER": "/nonexistent"},
        ):
            response = client.get("/q/health")
            assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "readiness" in data["checks"]
            assert "liveness" in data["checks"]


class TestHealthApp:
    """Test the dedicated health-only app (kubelet probe listener)."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for the health_app."""
        return TestClient(health_app)

    def test_readiness(self, client, tmp_path) -> None:
        """Test /q/health/ready endpoint on health_app."""
        with patch.dict(
            os.environ,
            {"SERVICE_STORAGE_FORMAT": "PVC", "STORAGE_DATA_FOLDER": str(tmp_path)},
        ):
            _health_cache.cache.clear()
            response = client.get("/q/health/ready")
            assert response.status_code in [
                HTTPStatus.OK,
                HTTPStatus.SERVICE_UNAVAILABLE,
            ]
            data = response.json()
            assert "status" in data
            assert "checks" in data

    def test_liveness(self, client) -> None:
        """Test /q/health/live endpoint on health_app."""
        response = client.get("/q/health/live")
        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert data["status"] == "alive"

    def test_general_health(self, client, tmp_path) -> None:
        """Test /q/health endpoint on health_app."""
        with patch.dict(
            os.environ,
            {"SERVICE_STORAGE_FORMAT": "PVC", "STORAGE_DATA_FOLDER": str(tmp_path)},
        ):
            _health_cache.cache.clear()
            response = client.get("/q/health")
            assert response.status_code in [
                HTTPStatus.OK,
                HTTPStatus.SERVICE_UNAVAILABLE,
            ]
            data = response.json()
            assert "checks" in data

    def test_no_other_routes(self, client) -> None:
        """Test that non-consumer routes are not exposed on health_app."""
        for path in ["/q/metrics", "/info"]:
            response = client.get(path)
            assert response.status_code == HTTPStatus.NOT_FOUND, (
                f"{path} should not exist on health app"
            )
        # "/" is a POST-only route (CloudEvent consumer), GET should be rejected
        response = client.get("/")
        assert response.status_code == HTTPStatus.METHOD_NOT_ALLOWED, (
            "GET / should be rejected on health app (POST only)"
        )

    def test_no_docs(self, client) -> None:
        """Test that OpenAPI docs are disabled on health_app."""
        for path in ["/docs", "/openapi.json"]:
            response = client.get(path)
            assert response.status_code == HTTPStatus.NOT_FOUND, (
                f"{path} should be disabled on health app"
            )
