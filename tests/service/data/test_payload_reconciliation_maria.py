"""
Tests for ModelMesh payload reconciliation with MariaDB storage.
"""

import unittest
import uuid

import pytest

pytest.importorskip("mariadb")
from src.service.data.modelmesh_parser import PartialPayload
from src.service.data.storage.maria.maria import MariaDBStorage
from tests.service.data.test_payload_reconciliation_pvc import TestPayloadReconciliation
from tests.service.data.test_utils import ModelMeshTestData


class TestMariaPayloadReconciliation(TestPayloadReconciliation):
    """
    Test class for ModelMesh payload reconciliation.
    """

    def setUp(self):
        """Set up the test environment."""

        self.storage = MariaDBStorage(
            "trustyai", "trustyai", "127.0.0.1", 3306, "trustyai-database", attempt_migration=False
        )

        self.model_name = "test-model"
        self.request_id = str(uuid.uuid4())

        input_specs = [("input", 5, 10, "INT32", 0, None)]
        output_specs = [("output", 5, 1, "INT32", 0)]

        self.input_payload_dict, self.output_payload_dict, _, _ = ModelMeshTestData.generate_test_payloads(
            self.model_name, input_specs, output_specs
        )

        self.input_payload = PartialPayload(**self.input_payload_dict)
        self.output_payload = PartialPayload(**self.output_payload_dict)

    def tearDown(self):
        """Clean up after tests."""
        self.storage.reset_database()


if __name__ == "__main__":
    unittest.main()
