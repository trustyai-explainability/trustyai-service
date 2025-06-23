"""
Tests for ModelMesh payload reconciliation.MariaDBStorage("root", "root", "127.0.0.1", 3306, "trustyai_database_v2")
"""

import asyncio
import unittest
import tempfile
import os
import base64
import time
from datetime import datetime
from unittest import mock
import uuid

import pandas as pd
import numpy as np

from src.service.data.modelmesh_parser import ModelMeshPayloadParser, PartialPayload
from src.service.data.storage.maria.maria import MariaDBStorage
from src.service.data.storage.pvc import PVCStorage
from tests.service.data.test_payload_reconciliation_pvc import TestPayloadReconciliation
from tests.service.data.test_utils import ModelMeshTestData


class TestMariaPayloadReconciliation(TestPayloadReconciliation):
    """
    Test class for ModelMesh payload reconciliation.
    """

    def setUp(self):
        """Set up the test environment."""

        self.storage = MariaDBStorage(
            "trustyai",
            "trustyai",
            "127.0.0.1",
            3306,
            "trustyai-database",
            attempt_migration=False
        )


        self.model_name = "test-model"
        self.request_id = str(uuid.uuid4())

        input_specs = [("input", 5, 10, "INT32", 0, None)]
        output_specs = [("output", 5, 1, "INT32", 0)]

        self.input_payload_dict, self.output_payload_dict, _, _ = (
            ModelMeshTestData.generate_test_payloads(
                self.model_name, input_specs, output_specs
            )
        )

        self.input_payload = PartialPayload(**self.input_payload_dict)
        self.output_payload = PartialPayload(**self.output_payload_dict)

    def tearDown(self):
        """Clean up after tests."""
        self.storage.reset_database()


if __name__ == "__main__":
    unittest.main()
