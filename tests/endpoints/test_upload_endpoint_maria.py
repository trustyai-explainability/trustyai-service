"""Tests for data upload endpoint with MariaDB storage."""

import asyncio
import os
import unittest
from importlib import reload

import pytest
from fastapi.testclient import TestClient

import trustyai_service.main
from tests.endpoints.test_upload_endpoint_pvc import TestUploadEndpointPVC


@pytest.mark.xdist_group("mariadb")
class TestUploadEndpointMaria(TestUploadEndpointPVC):
    """Test data upload endpoint with MariaDB storage backend."""

    def setUp(self) -> None:
        """Set up MariaDB storage environment and test client."""
        # Save original environment variables to restore in tearDown
        self.original_env = {
            "SERVICE_STORAGE_FORMAT": os.environ.get("SERVICE_STORAGE_FORMAT"),
            "DATABASE_USERNAME": os.environ.get("DATABASE_USERNAME"),
            "DATABASE_PASSWORD": os.environ.get("DATABASE_PASSWORD"),
            "DATABASE_HOST": os.environ.get("DATABASE_HOST"),
            "DATABASE_PORT": os.environ.get("DATABASE_PORT"),
            "DATABASE_DATABASE": os.environ.get("DATABASE_DATABASE"),
            "DATABASE_ATTEMPT_MIGRATION": os.environ.get("DATABASE_ATTEMPT_MIGRATION"),
        }

        # Use addCleanup to ensure environment is restored even if setUp fails
        def restore_environment() -> None:
            for key, value in self.original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore_environment)

        os.environ["SERVICE_STORAGE_FORMAT"] = "MARIA"
        os.environ["DATABASE_USERNAME"] = "trustyai"
        os.environ["DATABASE_PASSWORD"] = "trustyai"  # pragma: allowlist secret
        os.environ["DATABASE_HOST"] = "127.0.0.1"
        os.environ["DATABASE_PORT"] = "3306"
        os.environ["DATABASE_DATABASE"] = "trustyai-database"

        # Force reload of the global storage interface to use the new temp dir
        from trustyai_service.service.data import (  # noqa: PLC0415  # re-import after reload for test isolation
            storage,
        )

        self.storage_interface = storage.get_global_storage_interface(force_reload=True)

        # Re-create the FastAPI app to ensure it uses the new storage interface
        reload(trustyai_service.main)
        from trustyai_service.main import (  # noqa: PLC0415  # re-import after reload for test isolation
            app,
        )

        self.client = TestClient(app)

        self.original_datasets = set(
            asyncio.run(self.storage_interface.list_all_datasets()),
        )

    def tearDown(self) -> None:
        """Clean up test datasets and restore environment variables."""
        # delete any datasets we've created (only if setUp succeeded)
        if hasattr(self, "storage_interface") and hasattr(self, "original_datasets"):
            new_datasets = set(asyncio.run(self.storage_interface.list_all_datasets()))
            for ds in new_datasets.difference(self.original_datasets):
                asyncio.run(self.storage_interface.delete_dataset(ds))

        # Restore original environment variables (only if setUp saved them)
        if hasattr(self, "original_env"):
            for key, value in self.original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_upload_data(self) -> None:
        """Test data upload with MariaDB storage."""
        super().test_upload_data()

    def test_upload_multi_input_data(self) -> None:
        """Test multi-input data upload with MariaDB storage."""
        super().test_upload_multi_input_data()

    def test_upload_multi_input_data_no_unique_name(self) -> None:
        """Test multi-input data upload without unique names with MariaDB storage."""
        super().test_upload_multi_input_data_no_unique_name()

    def test_upload_multiple_tagging(self) -> None:
        """Test multiple tag upload with MariaDB storage."""
        super().test_upload_multiple_tagging()

    def test_upload_tag_that_uses_protected_name(self) -> None:
        """Test upload with protected tag names with MariaDB storage."""
        super().test_upload_tag_that_uses_protected_name()

    def test_upload_gaussian_data(self) -> None:
        """Test Gaussian data upload with MariaDB storage."""
        super().test_upload_gaussian_data()


if __name__ == "__main__":
    unittest.main()
