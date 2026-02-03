import asyncio
import os
import unittest

from fastapi.testclient import TestClient

from tests.endpoints.test_upload_endpoint_pvc import TestUploadEndpointPVC


class TestUploadEndpointMaria(TestUploadEndpointPVC):

    def setUp(self):
        os.environ["SERVICE_STORAGE_FORMAT"] = 'MARIA'
        os.environ["DATABASE_USERNAME"] = "trustyai"
        os.environ["DATABASE_PASSWORD"] = "trustyai"
        os.environ["DATABASE_HOST"] = "127.0.0.1"
        os.environ["DATABASE_PORT"] = "3306"
        os.environ["DATABASE_DATABASE"] = "trustyai-database"

        # Force reload of the global storage interface to use the new temp dir
        from src.service.data import storage
        self.storage_interface = storage.get_global_storage_interface(force_reload=True)

        # Re-create the FastAPI app to ensure it uses the new storage interface
        from importlib import reload
        import src.main
        reload(src.main)
        from src.main import app
        self.client = TestClient(app)

        self.original_datasets = set(asyncio.run(self.storage_interface.list_all_datasets()))

    def tearDown(self):
        # delete any datasets we've created
        new_datasets = set(asyncio.run(self.storage_interface.list_all_datasets()))
        for ds in new_datasets.difference(self.original_datasets):
            asyncio.run(self.storage_interface.delete_dataset(ds))

    def test_upload_data(self):
        super().test_upload_data()

    def test_upload_multi_input_data(self):
        super().test_upload_multi_input_data()

    def test_upload_multi_input_data_no_unique_name(self):
        super().test_upload_multi_input_data_no_unique_name()

    def test_upload_multiple_tagging(self):
        super().test_upload_multiple_tagging()

    def test_upload_tag_that_uses_protected_name(self):
        super().test_upload_tag_that_uses_protected_name()

    def test_upload_gaussian_data(self):
        super().test_upload_gaussian_data()


if __name__ == "__main__":
    unittest.main()