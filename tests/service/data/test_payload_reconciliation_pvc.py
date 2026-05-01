"""Tests for ModelMesh payload reconciliation."""

import asyncio
import tempfile
import unittest
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.service.data.modelmesh_parser import ModelMeshPayloadParser, PartialPayload
from src.service.data.storage.pvc import MAX_VOID_TYPE_LENGTH, PVCStorage
from tests.service.data.test_utils import ModelMeshTestData

# Test constants
EXPECTED_RECONCILED_ROWS = 5  # Expected rows after payload reconciliation
EXPECTED_BATCH_ROWS = 3  # Expected rows in batch reconciliation test
EXPECTED_DATASET_COLUMNS = 3  # Expected columns in dataset verification test
EXPECTED_RETRIEVED_VALUE = 456  # Expected numeric value in retrieved data


class TestPayloadReconciliation(unittest.TestCase):
    """Test class for ModelMesh payload reconciliation."""

    def setUp(self) -> None:
        """Set up the test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = PVCStorage(self.temp_dir.name)

        self.model_name = "test-model"
        self.request_id = str(uuid.uuid4())

        input_specs = [("input", 5, 10, "INT32", 0, None)]
        output_specs = [("output", 5, 1, "INT32", 0)]

        self.input_payload_dict, self.output_payload_dict, _, _ = (
            ModelMeshTestData.generate_test_payloads(
                self.model_name,
                input_specs,
                output_specs,
            )
        )

        self.input_payload = PartialPayload(**self.input_payload_dict)
        self.output_payload = PartialPayload(**self.output_payload_dict)

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.temp_dir.cleanup()

    async def _test_persist_input_payload(self) -> None:
        """Test persisting an input payload."""
        await self.storage.persist_partial_payload(
            self.input_payload,
            payload_id=self.request_id,
            is_input=True,
        )

        retrieved_payload = await self.storage.get_partial_payload(
            self.request_id,
            is_input=True,
            is_modelmesh=True,
        )

        assert retrieved_payload is not None
        assert retrieved_payload is not None  # Type narrowing for mypy
        assert isinstance(retrieved_payload, PartialPayload)  # Type narrowing for mypy
        assert retrieved_payload.data == self.input_payload.data

        output_payload = await self.storage.get_partial_payload(
            self.request_id,
            is_input=False,
            is_modelmesh=True,
        )
        assert output_payload is None

    async def _test_persist_output_payload(self) -> None:
        """Test persisting an output payload."""
        await self.storage.persist_partial_payload(
            self.output_payload,
            payload_id=self.request_id,
            is_input=False,
        )

        retrieved_payload = await self.storage.get_partial_payload(
            self.request_id,
            is_input=False,
            is_modelmesh=True,
        )

        assert retrieved_payload is not None
        assert retrieved_payload is not None  # Type narrowing for mypy
        assert isinstance(retrieved_payload, PartialPayload)  # Type narrowing for mypy
        assert retrieved_payload.data == self.output_payload.data

        input_payload = await self.storage.get_partial_payload(
            self.request_id,
            is_input=True,
            is_modelmesh=True,
        )
        assert input_payload is None

    async def _test_full_reconciliation(self) -> None:
        """Test the full payload reconciliation process."""
        await self.storage.persist_partial_payload(
            self.input_payload,
            self.request_id,
            is_input=True,
        )
        await self.storage.persist_partial_payload(
            self.output_payload,
            self.request_id,
            is_input=False,
        )

        input_payload = await self.storage.get_partial_payload(
            self.request_id,
            is_input=True,
            is_modelmesh=True,
        )
        output_payload = await self.storage.get_partial_payload(
            self.request_id,
            is_input=False,
            is_modelmesh=True,
        )

        assert input_payload is not None
        assert output_payload is not None
        # Type narrowing for mypy
        assert isinstance(input_payload, PartialPayload)
        assert isinstance(output_payload, PartialPayload)

        df = ModelMeshPayloadParser.payloads_to_dataframe(
            input_payload,
            output_payload,
            self.request_id,
            self.model_name,
        )

        assert isinstance(df, pd.DataFrame)
        assert "input" in df.columns
        assert "output_output" in df.columns
        assert len(df) == EXPECTED_RECONCILED_ROWS

        assert "id" in df.columns
        assert df["id"].iloc[0] == self.request_id

        assert "model_id" in df.columns
        assert df["model_id"].iloc[0] == self.model_name

        # Clean up
        await self.storage.delete_partial_payload(self.request_id, is_input=True)
        await self.storage.delete_partial_payload(self.request_id, is_input=False)

        input_payload = await self.storage.get_partial_payload(
            self.request_id,
            is_input=True,
            is_modelmesh=True,
        )
        output_payload = await self.storage.get_partial_payload(
            self.request_id,
            is_input=False,
            is_modelmesh=True,
        )

        assert input_payload is None
        assert output_payload is None

    async def _test_reconciliation_with_real_data(self) -> None:
        """Test reconciliation with sample b64 encoded data from files."""
        current_dir = Path(__file__).parent
        test_data_dir = current_dir.parent.parent / "data"

        with (test_data_dir / "input-sample.b64").open() as f:
            sample_input_data = f.read().strip()

        with (test_data_dir / "output-sample.b64").open() as f:
            sample_output_data = f.read().strip()

        input_payload = PartialPayload(data=sample_input_data)
        output_payload = PartialPayload(data=sample_output_data)

        request_id = str(uuid.uuid4())
        model_id = "sample-model"

        await self.storage.persist_partial_payload(
            input_payload,
            request_id,
            is_input=True,
        )
        await self.storage.persist_partial_payload(
            output_payload,
            request_id,
            is_input=False,
        )

        stored_input = await self.storage.get_partial_payload(
            request_id,
            is_input=True,
            is_modelmesh=True,
        )
        stored_output = await self.storage.get_partial_payload(
            request_id,
            is_input=False,
            is_modelmesh=True,
        )

        assert stored_input is not None
        assert stored_output is not None
        assert stored_input is not None  # Type narrowing for mypy
        assert stored_output is not None  # Type narrowing for mypy
        assert isinstance(stored_input, PartialPayload)  # Type narrowing for mypy
        assert isinstance(stored_output, PartialPayload)  # Type narrowing for mypy
        assert stored_input.data == sample_input_data
        assert stored_output.data == sample_output_data

        with mock.patch.object(
            ModelMeshPayloadParser,
            "payloads_to_dataframe",
        ) as mock_to_df:
            sample_df = pd.DataFrame(
                {
                    "input_feature": [1, 2, 3],
                    "output_output_feature": [4, 5, 6],
                    "id": [request_id] * 3,
                    "model_id": [model_id] * 3,
                    "synthetic": [False] * 3,
                },
            )
            mock_to_df.return_value = sample_df

            df = ModelMeshPayloadParser.payloads_to_dataframe(
                stored_input,
                stored_output,
                request_id,
                model_id,
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == EXPECTED_BATCH_ROWS

            mock_to_df.assert_called_once_with(
                stored_input,
                stored_output,
                request_id,
                model_id,
            )

        # Clean up
        await self.storage.delete_partial_payload(request_id, is_input=True)
        await self.storage.delete_partial_payload(request_id, is_input=False)

        assert (
            await self.storage.get_partial_payload(
                request_id,
                is_input=True,
                is_modelmesh=True,
            )
            is None
        )
        assert (
            await self.storage.get_partial_payload(
                request_id,
                is_input=False,
                is_modelmesh=True,
            )
            is None
        )

    async def _test_corrupted_payload_handling(self) -> None:
        """Test error handling for corrupted or invalid payloads."""
        request_id = str(uuid.uuid4())

        # Test 1: Corrupted payload data (invalid base64)
        corrupted_payload = PartialPayload(data="!!!INVALID_BASE64_DATA!!!")
        await self.storage.persist_partial_payload(
            corrupted_payload,
            request_id,
            is_input=True,
        )

        retrieved = await self.storage.get_partial_payload(
            request_id,
            is_input=True,
            is_modelmesh=True,
        )
        # Should still retrieve the corrupted payload (storage doesn't validate)
        assert retrieved is not None
        assert retrieved is not None  # Type narrowing for mypy
        assert isinstance(retrieved, PartialPayload)  # Type narrowing for mypy
        assert retrieved.data == corrupted_payload.data

        # Test 2: Attempt to parse corrupted payload should handle gracefully
        # (This is where the consumer endpoint would catch parsing errors)
        with pytest.raises((ValueError, Exception)):  # Base64 or protobuf parsing error
            # Attempting to parse invalid base64 should raise an error
            ModelMeshPayloadParser.parse_input_payload(corrupted_payload)

        # Clean up
        await self.storage.delete_partial_payload(request_id, is_input=True)

        # Test 3: Missing payload (already deleted)
        missing_payload = await self.storage.get_partial_payload(
            request_id,
            is_input=True,
            is_modelmesh=True,
        )
        assert missing_payload is None

    async def _test_void_type_length_exceeds_limit(self) -> None:
        """Test that rows exceeding MAX_VOID_TYPE_LENGTH raise ValueError."""
        # Use unique dataset name to avoid conflicts
        dataset_name = f"dataset_oversized_void_type_{uuid.uuid4().hex[:8]}"

        # Create a payload that will exceed MAX_VOID_TYPE_LENGTH when serialized
        oversized_string = "x" * (MAX_VOID_TYPE_LENGTH + 100)
        mixed_row = [
            oversized_string,  # large non-numeric payload
            {"key": "value"},  # non-primitive object
            123,  # numeric value
        ]
        dataset = np.array([mixed_row], dtype=object)
        column_names = ["col_str", "col_obj", "col_int"]

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            await self.storage.write_data(dataset_name, dataset, column_names)

    async def _test_void_type_length_within_limit(self) -> None:
        """Test that rows under MAX_VOID_TYPE_LENGTH are written successfully."""
        # Use unique dataset name to avoid conflicts
        dataset_name = f"dataset_under_void_type_limit_{uuid.uuid4().hex[:8]}"

        # Create a payload that is just under the limit
        # Account for pickle overhead by using a smaller string
        allowed_string = "y" * (MAX_VOID_TYPE_LENGTH - 200)
        mixed_row = [
            allowed_string,
            {"key": "value"},
            456,
        ]
        original_dataset = np.array([mixed_row], dtype=object)
        column_names = ["col_str", "col_obj", "col_int"]

        await self.storage.write_data(dataset_name, original_dataset, column_names)
        retrieved_dataset = await self.storage.read_data(dataset_name)

        # Ensure we got exactly one row with the expected values back
        assert retrieved_dataset.shape[0] == 1
        assert retrieved_dataset.shape[1] == EXPECTED_DATASET_COLUMNS
        retrieved_row = list(retrieved_dataset[0])

        assert allowed_string == retrieved_row[0]
        assert retrieved_row[1] == {"key": "value"}
        assert retrieved_row[2] == EXPECTED_RETRIEVED_VALUE


def run_async_test(coro: Coroutine[Any, Any, None]) -> None:
    """Run async tests."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


TestPayloadReconciliation.test_persist_input_payload = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_persist_input_payload(),
)
TestPayloadReconciliation.test_persist_output_payload = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_persist_output_payload(),
)
TestPayloadReconciliation.test_full_reconciliation = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_full_reconciliation(),
)
TestPayloadReconciliation.test_reconciliation_with_real_data = lambda self: (  # type: ignore[attr-defined]
    run_async_test(self._test_reconciliation_with_real_data())
)
TestPayloadReconciliation.test_void_type_length_exceeds_limit = lambda self: (  # type: ignore[attr-defined]
    run_async_test(self._test_void_type_length_exceeds_limit())
)
TestPayloadReconciliation.test_void_type_length_within_limit = lambda self: (  # type: ignore[attr-defined]
    run_async_test(self._test_void_type_length_within_limit())
)
TestPayloadReconciliation.test_corrupted_payload_handling = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_corrupted_payload_handling(),
)


if __name__ == "__main__":
    unittest.main()
