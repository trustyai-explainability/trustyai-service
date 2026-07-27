"""Tests for storage interface initialization with environment variables."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from trustyai_service.service.data.storage import get_storage_interface
from trustyai_service.service.data.storage.pvc import PVCStorage

# Test constants
DEFAULT_MARIADB_PORT = 3306
ALTERNATE_MARIADB_PORT = 3307

# Check if mariadb is available
try:
    import mariadb  # noqa: F401

    HAS_MARIADB = True
except ImportError:
    HAS_MARIADB = False


class TestStorageInterfaceEnvVars:
    """Test storage interface creation with different environment variable conventions."""

    def test_pvc_storage_creation(self) -> None:
        """Test PVC storage interface creation."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "PVC",
                "STORAGE_DATA_FOLDER": "/tmp/test",  # noqa: S108
                "STORAGE_DATA_FILENAME": "test.hdf5",
            },
            clear=False,
        ):
            storage = get_storage_interface()
            assert isinstance(storage, PVCStorage)
            assert storage.data_directory == "/tmp/test"  # noqa: S108
            assert storage.data_file == "test.hdf5"

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    @patch("trustyai_service.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_with_database_host_and_database(
        self, mock_storage: MagicMock
    ) -> None:
        """Test MariaDB with DATABASE_HOST and DATABASE_DATABASE (direct deployment)."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "MARIA",
                "DATABASE_USERNAME": "test_user",
                "DATABASE_PASSWORD": "test_pass",  # pragma: allowlist secret
                "DATABASE_HOST": "localhost",
                "DATABASE_PORT": str(DEFAULT_MARIADB_PORT),
                "DATABASE_DATABASE": "test_db",
                "DATABASE_ATTEMPT_MIGRATION": "0",
            },
            clear=False,
        ):
            get_storage_interface()
            # Verify MariaDBStorage was called with correct env vars
            mock_storage.assert_called_once_with(
                user="test_user",
                password="test_pass",  # noqa: S106  # pragma: allowlist secret
                host="localhost",
                port=DEFAULT_MARIADB_PORT,
                database="test_db",
                ssl_ca=None,
                attempt_migration=False,
            )

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    @patch("trustyai_service.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_with_database_service_and_name(
        self, mock_storage: MagicMock
    ) -> None:
        """Test MariaDB with DATABASE_SERVICE and DATABASE_NAME (operator deployment)."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "MARIA",
                "DATABASE_USERNAME": "operator_user",
                "DATABASE_PASSWORD": "operator_pass",  # pragma: allowlist secret
                "DATABASE_SERVICE": "mariadb-service",
                "DATABASE_PORT": str(ALTERNATE_MARIADB_PORT),
                "DATABASE_NAME": "operator_db",
                "DATABASE_ATTEMPT_MIGRATION": "false",
            },
            clear=False,
        ):
            get_storage_interface()
            # Verify MariaDBStorage was called with operator env vars
            mock_storage.assert_called_once_with(
                user="operator_user",
                password="operator_pass",  # noqa: S106  # pragma: allowlist secret
                host="mariadb-service",
                port=ALTERNATE_MARIADB_PORT,
                database="operator_db",
                ssl_ca=None,
                attempt_migration=False,
            )

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    @patch("trustyai_service.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_fallback_priority(self, mock_storage: MagicMock) -> None:
        """Test that DATABASE_HOST takes priority over DATABASE_SERVICE when both present."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "MARIA",
                "DATABASE_USERNAME": "test_user",
                "DATABASE_PASSWORD": "test_pass",  # pragma: allowlist secret
                "DATABASE_HOST": "direct_host",
                "DATABASE_SERVICE": "operator_host",
                "DATABASE_DATABASE": "direct_db",
                "DATABASE_NAME": "operator_db",
                "DATABASE_PORT": str(DEFAULT_MARIADB_PORT),
                "DATABASE_ATTEMPT_MIGRATION": "0",
            },
            clear=False,
        ):
            get_storage_interface()
            # Verify DATABASE_HOST and DATABASE_DATABASE take priority
            mock_storage.assert_called_once_with(
                user="test_user",
                password="test_pass",  # noqa: S106  # pragma: allowlist secret
                host="direct_host",
                port=DEFAULT_MARIADB_PORT,
                database="direct_db",
                ssl_ca=None,
                attempt_migration=False,
            )

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    @patch("trustyai_service.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_mixed_conventions(self, mock_storage: MagicMock) -> None:
        """Test MariaDB with mixed env var conventions (DATABASE_HOST + DATABASE_NAME)."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "MARIA",
                "DATABASE_USERNAME": "test_user",
                "DATABASE_PASSWORD": "test_pass",  # pragma: allowlist secret
                "DATABASE_HOST": "mixed_host",
                "DATABASE_NAME": "mixed_db",
                "DATABASE_PORT": str(DEFAULT_MARIADB_PORT),
                "DATABASE_ATTEMPT_MIGRATION": "0",
            },
            clear=False,
        ):
            get_storage_interface()
            # Verify mixed conventions work
            mock_storage.assert_called_once_with(
                user="test_user",
                password="test_pass",  # noqa: S106  # pragma: allowlist secret
                host="mixed_host",
                port=DEFAULT_MARIADB_PORT,
                database="mixed_db",
                ssl_ca=None,
                attempt_migration=False,
            )

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    @patch("trustyai_service.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_with_database_format(self, mock_storage: MagicMock) -> None:
        """Test MariaDB with SERVICE_STORAGE_FORMAT=DATABASE (operator convention)."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "DATABASE",
                "DATABASE_USERNAME": "operator_user",
                "DATABASE_PASSWORD": "operator_pass",  # pragma: allowlist secret
                "DATABASE_SERVICE": "mariadb-service",
                "DATABASE_PORT": str(DEFAULT_MARIADB_PORT),
                "DATABASE_NAME": "operator_db",
                "DATABASE_ATTEMPT_MIGRATION": "0",
            },
            clear=False,
        ):
            get_storage_interface()
            # Verify DATABASE storage format works (operator convention)
            mock_storage.assert_called_once_with(
                user="operator_user",
                password="operator_pass",  # noqa: S106  # pragma: allowlist secret
                host="mariadb-service",
                port=DEFAULT_MARIADB_PORT,
                database="operator_db",
                ssl_ca=None,
                attempt_migration=False,
            )

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    @patch("trustyai_service.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_with_quarkus_credentials(self, mock_storage: MagicMock) -> None:
        """Test MariaDB with QUARKUS_DATASOURCE_USERNAME/PASSWORD (operator convention)."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "DATABASE",
                "DATABASE_USERNAME": "",  # Explicitly clear to test fallback
                "DATABASE_PASSWORD": "",  # Explicitly clear to test fallback
                "QUARKUS_DATASOURCE_USERNAME": "quarkus_user",
                "QUARKUS_DATASOURCE_PASSWORD": "quarkus_pass",  # pragma: allowlist secret
                "DATABASE_SERVICE": "mariadb-service",
                "DATABASE_PORT": str(DEFAULT_MARIADB_PORT),
                "DATABASE_NAME": "operator_db",
                "DATABASE_ATTEMPT_MIGRATION": "0",
            },
            clear=False,
        ):
            get_storage_interface()
            # Verify QUARKUS_DATASOURCE_* env vars work
            mock_storage.assert_called_once_with(
                user="quarkus_user",
                password="quarkus_pass",  # noqa: S106  # pragma: allowlist secret
                host="mariadb-service",
                port=DEFAULT_MARIADB_PORT,
                database="operator_db",
                ssl_ca=None,
                attempt_migration=False,
            )

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    @patch("trustyai_service.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_credentials_fallback_priority(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that DATABASE_USERNAME/PASSWORD take priority over QUARKUS_DATASOURCE_*."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "MARIA",
                "DATABASE_USERNAME": "direct_user",
                "QUARKUS_DATASOURCE_USERNAME": "quarkus_user",
                "DATABASE_PASSWORD": "direct_pass",  # pragma: allowlist secret
                "QUARKUS_DATASOURCE_PASSWORD": "quarkus_pass",  # pragma: allowlist secret
                "DATABASE_HOST": "localhost",
                "DATABASE_PORT": str(DEFAULT_MARIADB_PORT),
                "DATABASE_DATABASE": "test_db",
                "DATABASE_ATTEMPT_MIGRATION": "0",
            },
            clear=False,
        ):
            get_storage_interface()
            # Verify DATABASE_USERNAME/PASSWORD take priority
            mock_storage.assert_called_once_with(
                user="direct_user",
                password="direct_pass",  # noqa: S106  # pragma: allowlist secret
                host="localhost",
                port=DEFAULT_MARIADB_PORT,
                database="test_db",
                ssl_ca=None,
                attempt_migration=False,
            )

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    @patch("trustyai_service.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_ssl_ca_passed_when_file_exists(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that ssl_ca is passed through when the CA cert file exists."""
        with (
            tempfile.NamedTemporaryFile(suffix=".crt") as ca_file,
            patch.dict(
                os.environ,
                {
                    "SERVICE_STORAGE_FORMAT": "MARIA",
                    "DATABASE_USERNAME": "test_user",
                    "DATABASE_PASSWORD": "test_pass",  # pragma: allowlist secret
                    "DATABASE_HOST": "localhost",
                    "DATABASE_PORT": str(DEFAULT_MARIADB_PORT),
                    "DATABASE_DATABASE": "test_db",
                    "DATABASE_TLS_CA_CERT": ca_file.name,
                    "DATABASE_ATTEMPT_MIGRATION": "0",
                },
                clear=False,
            ),
        ):
            get_storage_interface()
            mock_storage.assert_called_once_with(
                user="test_user",
                password="test_pass",  # noqa: S106  # pragma: allowlist secret
                host="localhost",
                port=DEFAULT_MARIADB_PORT,
                database="test_db",
                ssl_ca=ca_file.name,
                attempt_migration=False,
            )

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    @patch("trustyai_service.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_ssl_ca_none_when_file_missing(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that ssl_ca is None when the configured CA cert file does not exist."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "MARIA",
                "DATABASE_USERNAME": "test_user",
                "DATABASE_PASSWORD": "test_pass",  # pragma: allowlist secret
                "DATABASE_HOST": "localhost",
                "DATABASE_PORT": str(DEFAULT_MARIADB_PORT),
                "DATABASE_DATABASE": "test_db",
                "DATABASE_TLS_CA_CERT": "/nonexistent/path/ca.crt",
                "DATABASE_ATTEMPT_MIGRATION": "0",
            },
            clear=False,
        ):
            get_storage_interface()
            mock_storage.assert_called_once_with(
                user="test_user",
                password="test_pass",  # noqa: S106  # pragma: allowlist secret
                host="localhost",
                port=DEFAULT_MARIADB_PORT,
                database="test_db",
                ssl_ca=None,
                attempt_migration=False,
            )

    def test_unsupported_storage_format(self) -> None:
        """Test that unsupported storage format raises ValueError."""
        with (
            patch.dict(
                os.environ, {"SERVICE_STORAGE_FORMAT": "UNSUPPORTED"}, clear=False
            ),
            pytest.raises(ValueError, match="not yet supported"),
        ):
            get_storage_interface()

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    def test_mariadb_missing_all_parameters(self) -> None:
        """Test that missing all MariaDB parameters raises ValueError."""
        with (
            patch.dict(
                os.environ,
                {"SERVICE_STORAGE_FORMAT": "MARIA"},
                clear=True,
            ),
            pytest.raises(
                ValueError,
                match="MariaDB storage requires environment variables: "
                "DATABASE_USERNAME or QUARKUS_DATASOURCE_USERNAME, "
                "DATABASE_PASSWORD or QUARKUS_DATASOURCE_PASSWORD, "
                "DATABASE_HOST or DATABASE_SERVICE, "
                "DATABASE_DATABASE or DATABASE_NAME",
            ),
        ):
            get_storage_interface()

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    def test_mariadb_missing_username(self) -> None:
        """Test that missing username raises ValueError."""
        with (
            patch.dict(
                os.environ,
                {
                    "SERVICE_STORAGE_FORMAT": "MARIA",
                    "DATABASE_PASSWORD": "test_pass",  # pragma: allowlist secret
                    "DATABASE_HOST": "localhost",
                    "DATABASE_DATABASE": "test_db",
                },
                clear=True,
            ),
            pytest.raises(
                ValueError,
                match="MariaDB storage requires environment variables: "
                "DATABASE_USERNAME or QUARKUS_DATASOURCE_USERNAME",
            ),
        ):
            get_storage_interface()

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    def test_mariadb_missing_password(self) -> None:
        """Test that missing password raises ValueError."""
        with (
            patch.dict(
                os.environ,
                {
                    "SERVICE_STORAGE_FORMAT": "MARIA",
                    "DATABASE_USERNAME": "test_user",
                    "DATABASE_HOST": "localhost",
                    "DATABASE_DATABASE": "test_db",
                },
                clear=True,
            ),
            pytest.raises(
                ValueError,
                match="MariaDB storage requires environment variables: "
                "DATABASE_PASSWORD or QUARKUS_DATASOURCE_PASSWORD",
            ),
        ):
            get_storage_interface()

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    def test_mariadb_missing_host(self) -> None:
        """Test that missing host raises ValueError."""
        with (
            patch.dict(
                os.environ,
                {
                    "SERVICE_STORAGE_FORMAT": "MARIA",
                    "DATABASE_USERNAME": "test_user",
                    "DATABASE_PASSWORD": "test_pass",  # pragma: allowlist secret
                    "DATABASE_DATABASE": "test_db",
                },
                clear=True,
            ),
            pytest.raises(
                ValueError,
                match="MariaDB storage requires environment variables: "
                "DATABASE_HOST or DATABASE_SERVICE",
            ),
        ):
            get_storage_interface()

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    def test_mariadb_missing_database(self) -> None:
        """Test that missing database raises ValueError."""
        with (
            patch.dict(
                os.environ,
                {
                    "SERVICE_STORAGE_FORMAT": "MARIA",
                    "DATABASE_USERNAME": "test_user",
                    "DATABASE_PASSWORD": "test_pass",  # pragma: allowlist secret
                    "DATABASE_HOST": "localhost",
                },
                clear=True,
            ),
            pytest.raises(
                ValueError,
                match="MariaDB storage requires environment variables: "
                "DATABASE_DATABASE or DATABASE_NAME",
            ),
        ):
            get_storage_interface()

    @pytest.mark.skipif(not HAS_MARIADB, reason="mariadb extra not installed")
    def test_mariadb_missing_multiple_parameters(self) -> None:
        """Test that missing multiple MariaDB parameters raises ValueError with all missing listed."""
        with (
            patch.dict(
                os.environ,
                {
                    "SERVICE_STORAGE_FORMAT": "MARIA",
                    "DATABASE_USERNAME": "test_user",
                    "DATABASE_HOST": "localhost",
                },
                clear=True,
            ),
            pytest.raises(
                ValueError,
                match="MariaDB storage requires environment variables: "
                "DATABASE_PASSWORD or QUARKUS_DATASOURCE_PASSWORD, "
                "DATABASE_DATABASE or DATABASE_NAME",
            ),
        ):
            get_storage_interface()
