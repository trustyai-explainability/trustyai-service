"""Tests for storage backend factory."""

import os
from unittest.mock import patch

import pytest

from trustyai_service.service.data.storage import get_storage_interface
from trustyai_service.service.data.storage.pvc import PVCStorage


class TestGetStorageInterface:
    """Tests for get_storage_interface factory function."""

    def test_pvc_is_default(self) -> None:
        """PVC storage is returned when no format is specified."""
        env = os.environ.copy()
        env.pop("SERVICE_STORAGE_FORMAT", None)
        with patch.dict(os.environ, env, clear=True):
            storage = get_storage_interface()
            assert isinstance(storage, PVCStorage)

    def test_unsupported_format_raises(self) -> None:
        """Unsupported format raises ValueError."""
        with (
            patch.dict(os.environ, {"SERVICE_STORAGE_FORMAT": "REDIS"}, clear=False),
            pytest.raises(ValueError, match="not yet supported"),
        ):
            get_storage_interface()


@pytest.mark.skipif(
    not pytest.importorskip("mariadb", reason="mariadb not installed"),
    reason="mariadb not installed",
)
class TestGetStorageInterfaceMariaDB:
    """Tests for MariaDB storage format routing (requires mariadb package)."""

    @patch(
        "trustyai_service.service.data.storage.maria.maria.MariaDBStorage.__init__",
        return_value=None,
    )
    def test_maria_format_creates_mariadb(self, _mock_init: object) -> None:
        """MARIA format returns MariaDBStorage."""
        from trustyai_service.service.data.storage.maria.maria import (  # noqa: PLC0415
            MariaDBStorage,
        )

        env = {
            "SERVICE_STORAGE_FORMAT": "MARIA",
            "DATABASE_USERNAME": "user",
            "DATABASE_PASSWORD": "pass",  # pragma: allowlist secret
            "DATABASE_HOST": "localhost",
            "DATABASE_PORT": "3306",
            "DATABASE_DATABASE": "testdb",
        }
        with patch.dict(os.environ, env, clear=False):
            storage = get_storage_interface()
            assert isinstance(storage, MariaDBStorage)

    @patch(
        "trustyai_service.service.data.storage.maria.maria.MariaDBStorage.__init__",
        return_value=None,
    )
    def test_database_format_creates_mariadb(self, _mock_init: object) -> None:
        """DATABASE format is accepted as alias for MARIA."""
        from trustyai_service.service.data.storage.maria.maria import (  # noqa: PLC0415
            MariaDBStorage,
        )

        env = {
            "SERVICE_STORAGE_FORMAT": "DATABASE",
            "DATABASE_USERNAME": "user",
            "DATABASE_PASSWORD": "pass",  # pragma: allowlist secret
            "DATABASE_HOST": "localhost",
            "DATABASE_PORT": "3306",
            "DATABASE_DATABASE": "testdb",
        }
        with patch.dict(os.environ, env, clear=False):
            storage = get_storage_interface()
            assert isinstance(storage, MariaDBStorage)
