"""Tests for storage interface initialization with environment variables."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.service.data.storage import get_storage_interface
from src.service.data.storage.pvc import PVCStorage

# Test constants
DEFAULT_MARIADB_PORT = 3306
ALTERNATE_MARIADB_PORT = 3307


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

    @pytest.mark.skipif(
        not pytest.importorskip("mariadb", reason="mariadb extra not installed"),
        reason="mariadb extra not installed",
    )
    @patch("src.service.data.storage.maria.maria.MariaDBStorage")
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
                attempt_migration=False,
            )

    @pytest.mark.skipif(
        not pytest.importorskip("mariadb", reason="mariadb extra not installed"),
        reason="mariadb extra not installed",
    )
    @patch("src.service.data.storage.maria.maria.MariaDBStorage")
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
                attempt_migration=False,
            )

    @pytest.mark.skipif(
        not pytest.importorskip("mariadb", reason="mariadb extra not installed"),
        reason="mariadb extra not installed",
    )
    @patch("src.service.data.storage.maria.maria.MariaDBStorage")
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
                attempt_migration=False,
            )

    @pytest.mark.skipif(
        not pytest.importorskip("mariadb", reason="mariadb extra not installed"),
        reason="mariadb extra not installed",
    )
    @patch("src.service.data.storage.maria.maria.MariaDBStorage")
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
                attempt_migration=False,
            )

    @pytest.mark.skipif(
        not pytest.importorskip("mariadb", reason="mariadb extra not installed"),
        reason="mariadb extra not installed",
    )
    @patch("src.service.data.storage.maria.maria.MariaDBStorage")
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
                attempt_migration=False,
            )

    @pytest.mark.skipif(
        not pytest.importorskip("mariadb", reason="mariadb extra not installed"),
        reason="mariadb extra not installed",
    )
    @patch("src.service.data.storage.maria.maria.MariaDBStorage")
    def test_mariadb_with_quarkus_credentials(self, mock_storage: MagicMock) -> None:
        """Test MariaDB with QUARKUS_DATASOURCE_USERNAME/PASSWORD (operator convention)."""
        with patch.dict(
            os.environ,
            {
                "SERVICE_STORAGE_FORMAT": "DATABASE",
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
                attempt_migration=False,
            )

    @pytest.mark.skipif(
        not pytest.importorskip("mariadb", reason="mariadb extra not installed"),
        reason="mariadb extra not installed",
    )
    @patch("src.service.data.storage.maria.maria.MariaDBStorage")
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
