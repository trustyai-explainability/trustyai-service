"""
Tests for the ModelMesh protobuf parser.
"""

import unittest

import pandas as pd

from src.service.data.modelmesh_parser import ModelMeshPayloadParser, PartialPayload
from tests.service.data.test_utils import ModelMeshTestData


class TestModelMeshParser(unittest.TestCase):
    """
    Test the ModelMeshPayloadParser.
    """

    def test_standardize_model_id(self):
        """Test the standardize_model_id method."""
        model_id = "my-model__isvc-123456"
        self.assertEqual(ModelMeshPayloadParser.standardize_model_id(model_id), "my-model")

        model_id = "my-model"
        self.assertEqual(ModelMeshPayloadParser.standardize_model_id(model_id), "my-model")

        self.assertEqual(ModelMeshPayloadParser.standardize_model_id(None), None)

    def test_parse_input_payload(self):
        """Test parsing of input payload."""
        model_name = "test-model"
        input_specs = [("input", 5, 10, "INT32", 0, None)]
        output_specs = [("output", 5, 1, "INT32", 0)]

        input_payload_dict, _, _, _ = ModelMeshTestData.generate_test_payloads(model_name, input_specs, output_specs)

        input_payload = PartialPayload(**input_payload_dict)

        request = ModelMeshPayloadParser.parse_input_payload(input_payload)

        self.assertEqual(request.model_name, model_name)
        self.assertEqual(len(request.inputs), 1)

        tensor = request.inputs[0]
        self.assertEqual(tensor.name, "input")
        self.assertEqual(tensor.datatype, "INT32")
        self.assertEqual(list(tensor.shape), [5, 10])

    def test_parse_output_payload(self):
        """Test parsing of output payload."""
        model_name = "test-model"
        input_specs = [("input", 5, 10, "INT32", 0, None)]
        output_specs = [("output", 5, 1, "INT32", 0)]

        _, output_payload_dict, _, _ = ModelMeshTestData.generate_test_payloads(model_name, input_specs, output_specs)

        output_payload = PartialPayload(**output_payload_dict)

        response = ModelMeshPayloadParser.parse_output_payload(output_payload)

        self.assertTrue(response.model_name.startswith(model_name))
        self.assertEqual(response.model_version, "1")
        self.assertEqual(len(response.outputs), 1)

        tensor = response.outputs[0]
        self.assertEqual(tensor.name, "output")
        self.assertEqual(tensor.datatype, "INT32")
        self.assertEqual(list(tensor.shape), [5, 1])

    def test_payloads_to_dataframe_int32(self):
        """Test converting payloads to DataFrame with INT32 data."""
        self._test_payloads_to_dataframe("INT32")

    def test_payloads_to_dataframe_fp64(self):
        """Test converting payloads to DataFrame with FP64 data."""
        self._test_payloads_to_dataframe("FP64")

    def test_payloads_to_dataframe_bool(self):
        """Test converting payloads to DataFrame with BOOL data."""
        self._test_payloads_to_dataframe("BOOL")

    def test_payloads_to_dataframe_with_synthetic_flag(self):
        """Test handling of synthetic data flag."""
        model_name = "test-model"
        bias_ignore_params = {"bias_ignore": "true"}
        input_specs = [("input", 5, 10, "INT32", 0, bias_ignore_params)]
        output_specs = [("output", 5, 1, "INT32", 0)]

        input_payload_dict, output_payload_dict, input_data, output_data = ModelMeshTestData.generate_test_payloads(
            model_name, input_specs, output_specs
        )

        input_payload = PartialPayload(**input_payload_dict)
        output_payload = PartialPayload(**output_payload_dict)

        df = ModelMeshPayloadParser.payloads_to_dataframe(input_payload, output_payload, "test-id", model_name)

        self.assertTrue(all(df["synthetic"]))

    def _test_payloads_to_dataframe(self, datatype: str):
        """Helper method to test payloads_to_dataframe with different datatypes."""
        model_name = "test-model"
        n_rows, n_input_cols, n_output_cols = 5, 3, 2

        input_specs = [(f"input_{i}", n_rows, 1, datatype, i, None) for i in range(n_input_cols)]

        output_specs = [(f"output_{i}", n_rows, 1, datatype, i) for i in range(n_output_cols)]

        input_payload_dict, output_payload_dict, input_data, output_data = ModelMeshTestData.generate_test_payloads(
            model_name, input_specs, output_specs
        )

        input_payload = PartialPayload(**input_payload_dict)
        output_payload = PartialPayload(**output_payload_dict)

        df = ModelMeshPayloadParser.payloads_to_dataframe(input_payload, output_payload, "test-id", model_name)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), n_rows)

        for i in range(n_input_cols):
            col_name = f"input_{i}"
            self.assertIn(col_name, df.columns)

        for i in range(n_output_cols):
            col_name = f"output_output_{i}"
            self.assertIn(col_name, df.columns)

        self.assertIn("id", df.columns)
        self.assertEqual(df["id"].iloc[0], "test-id")

        self.assertIn("model_id", df.columns)
        self.assertEqual(df["model_id"].iloc[0], model_name)

        self.assertIn("synthetic", df.columns)
        self.assertFalse(df["synthetic"].iloc[0])

        self.assertEqual(df.attrs["input_tensor_name"], "input_0")
        self.assertEqual(df.attrs["output_tensor_name"], "output_0")


if __name__ == "__main__":
    unittest.main()
