"""Tests for reconcile_kserve tag assignment and write_reconciled_data.

Covers: default unlabeled tag, custom tag, synthetic tag via bias-ignore,
metadata update failure warning, and write_reconciled_data storage calls.
"""

import asyncio
import unittest
from http import HTTPStatus
from typing import Self
from unittest import mock

import numpy as np
import pytest
from fastapi import HTTPException

from trustyai_service.endpoints.consumer import (
    KServeData,
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.endpoints.consumer.consumer_endpoint import (
    BIAS_IGNORE_PARAM,
    SYNTHETIC_TAG,
    UNLABELED_TAG,
    _validate_payload_type,
    reconcile_kserve,
    write_reconciled_data,
)

# Expected call counts for write_reconciled_data (inputs + outputs + metadata)
_EXPECTED_DATASET_WRITES = 3
# Expected call counts for partial payload cleanup (input + output)
_EXPECTED_PARTIAL_DELETES = 2


def _simple_request(
    *,
    id_: str = "r-1",
    parameters: dict[str, str] | None = None,
) -> KServeInferenceRequest:
    """Build a minimal KServeInferenceRequest with 3 rows."""
    return KServeInferenceRequest(
        id=id_,
        parameters=parameters,
        inputs=[
            KServeData(
                name="input",
                shape=[3],
                datatype="FP32",
                data=[1.0, 2.0, 3.0],
            ),
        ],
    )


def _simple_response(
    *,
    id_: str = "r-1",
    model_name: str = "test-model",
) -> KServeInferenceResponse:
    """Build a minimal KServeInferenceResponse with 3 rows."""
    return KServeInferenceResponse(
        id=id_,
        model_name=model_name,
        outputs=[
            KServeData(
                name="output",
                shape=[3],
                datatype="FP32",
                data=[10.0, 20.0, 30.0],
            ),
        ],
    )


def _run(coro: object) -> None:
    """Run an async coroutine synchronously via a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


class TestReconcileKserveTagAssignment(unittest.TestCase):
    """Test tag assignment logic in reconcile_kserve."""

    def setUp(self) -> None:
        """Patch write_reconciled_data so we can inspect tag arguments."""
        self.write_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.write_reconciled_data",
            new=mock.AsyncMock(),
        )
        self.mock_write = self.write_patch.start()

    def tearDown(self) -> None:
        """Stop write_reconciled_data patch."""
        self.write_patch.stop()

    def test_default_tag_is_unlabeled(self) -> None:
        """Without explicit tag or bias-ignore, tag should be UNLABELED."""
        req = _simple_request()
        resp = _simple_response()

        _run(reconcile_kserve(req, resp, tag=None))

        call_kwargs = self.mock_write.call_args[1]
        assert call_kwargs["tags"] == [UNLABELED_TAG]

    def test_custom_tag_passed_through(self) -> None:
        """An explicit tag parameter is forwarded to write_reconciled_data."""
        req = _simple_request()
        resp = _simple_response()

        _run(reconcile_kserve(req, resp, tag="TRAINING"))

        call_kwargs = self.mock_write.call_args[1]
        assert call_kwargs["tags"] == ["TRAINING"]

    def test_bias_ignore_sets_synthetic_tag(self) -> None:
        """bias-ignore=true in parameters sets the SYNTHETIC tag."""
        req = _simple_request(
            parameters={BIAS_IGNORE_PARAM: "true"},
        )
        resp = _simple_response()

        _run(reconcile_kserve(req, resp, tag=None))

        call_kwargs = self.mock_write.call_args[1]
        assert call_kwargs["tags"] == [SYNTHETIC_TAG]

    def test_bias_ignore_false_keeps_unlabeled(self) -> None:
        """bias-ignore=false (or any non-true value) keeps UNLABELED tag."""
        req = _simple_request(
            parameters={BIAS_IGNORE_PARAM: "false"},
        )
        resp = _simple_response()

        _run(reconcile_kserve(req, resp, tag=None))

        call_kwargs = self.mock_write.call_args[1]
        assert call_kwargs["tags"] == [UNLABELED_TAG]

    def test_explicit_tag_overrides_bias_ignore(self) -> None:
        """An explicit tag takes precedence over bias-ignore parameter."""
        req = _simple_request(
            parameters={BIAS_IGNORE_PARAM: "true"},
        )
        resp = _simple_response()

        _run(reconcile_kserve(req, resp, tag="CUSTOM"))

        call_kwargs = self.mock_write.call_args[1]
        assert call_kwargs["tags"] == ["CUSTOM"]

    def test_no_parameters_defaults_to_unlabeled(self) -> None:
        """Request with parameters=None defaults to UNLABELED."""
        req = _simple_request(parameters=None)
        resp = _simple_response()

        _run(reconcile_kserve(req, resp, tag=None))

        call_kwargs = self.mock_write.call_args[1]
        assert call_kwargs["tags"] == [UNLABELED_TAG]

    def test_model_name_from_response(self) -> None:
        """model_id is taken from the response's model_name."""
        req = _simple_request()
        resp = _simple_response(model_name="my-model")

        _run(reconcile_kserve(req, resp, tag=None))

        call_kwargs = self.mock_write.call_args[1]
        assert call_kwargs["model_id"] == "my-model"

    def test_request_id_forwarded(self) -> None:
        """The request id is forwarded to write_reconciled_data."""
        req = _simple_request(id_="unique-id-123")
        resp = _simple_response()

        _run(reconcile_kserve(req, resp, tag=None))

        call_kwargs = self.mock_write.call_args[1]
        assert call_kwargs["id_"] == "unique-id-123"


class TestWriteReconciledData(unittest.TestCase):
    """Test write_reconciled_data storage interactions."""

    def setUp(self) -> None:
        """Patch storage, data source, and ModelData."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.data_source_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_data_source",
        )
        self.model_data_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.ModelData",
        )

        self.mock_storage = self.storage_patch.start().return_value
        self.mock_storage.write_data = mock.AsyncMock()
        self.mock_storage.delete_partial_payload = mock.AsyncMock()

        self.mock_data_source = self.data_source_patch.start().return_value
        self.mock_data_source.add_model_to_known = mock.AsyncMock()
        self.mock_data_source.get_known_models = mock.AsyncMock(
            return_value={"test-model"}
        )
        self.mock_metadata = mock.MagicMock()
        self.mock_data_source.get_metadata = mock.AsyncMock(
            return_value=self.mock_metadata
        )

        self.mock_model_data_cls = self.model_data_patch.start()
        self.mock_model_data_cls.return_value.shapes = mock.AsyncMock(
            return_value=[(3, 1), (3, 1), (3, 4)]
        )

    def tearDown(self) -> None:
        """Stop storage, data source, and ModelData patches."""
        self.storage_patch.stop()
        self.data_source_patch.stop()
        self.model_data_patch.stop()

    def test_writes_three_datasets(self) -> None:
        """write_reconciled_data calls write_data for inputs, outputs, metadata."""
        input_arr = np.array([[1.0], [2.0], [3.0]])
        output_arr = np.array([[10.0], [20.0], [30.0]])

        _run(
            write_reconciled_data(
                input_arr,
                ["input"],
                output_arr,
                ["output"],
                model_id="test-model",
                tags=["TRAINING"],
                id_="wr-1",
            )
        )

        assert self.mock_storage.write_data.call_count == _EXPECTED_DATASET_WRITES
        dataset_names = [
            call[0][0] for call in self.mock_storage.write_data.call_args_list
        ]
        assert "test-model_inputs" in dataset_names
        assert "test-model_outputs" in dataset_names
        assert "test-model_metadata" in dataset_names

    def test_writes_three_datasets_under_model_lock(self) -> None:
        """Keep grouped dataset writes inside the model coordination lock."""
        events: list[str] = []

        class RecordingLock:
            async def __aenter__(self) -> Self:
                events.append("enter")
                return self

            async def __aexit__(self, *_args: object) -> None:
                events.append("exit")

        async def write_data(*_args: object, **_kwargs: object) -> None:
            events.append("write")

        self.mock_storage.write_data.side_effect = write_data
        with mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_model_lock",
            return_value=RecordingLock(),
        ):
            _run(
                write_reconciled_data(
                    np.array([[1.0]]),
                    ["input"],
                    np.array([[2.0]]),
                    ["output"],
                    model_id="test-model",
                    tags=["TRAINING"],
                    id_="lock-1",
                )
            )

        assert events[0] == "enter"
        assert events[-1] == "exit"
        assert "write" in events[1:-1]

    def test_model_lock_waits_for_cancelled_sibling_writes(self) -> None:
        """Keep the model lock until all grouped writes finish cancellation."""
        events: list[str] = []
        writes_started = asyncio.Event()
        started_count = 0

        class RecordingLock:
            async def __aenter__(self) -> Self:
                events.append("enter")
                return self

            async def __aexit__(self, *_args: object) -> None:
                events.append("exit")

        async def write_data(dataset_name: str, *_args: object) -> None:
            nonlocal started_count
            started_count += 1
            if started_count == _EXPECTED_DATASET_WRITES:
                writes_started.set()
            await writes_started.wait()
            try:
                if dataset_name.endswith("_inputs"):
                    failure_message = "input write failed"
                    raise RuntimeError(failure_message)
                await asyncio.sleep(0)
            finally:
                events.append(f"finished:{dataset_name}")

        self.mock_storage.write_data.side_effect = write_data
        with (
            mock.patch(
                "trustyai_service.endpoints.consumer.consumer_endpoint.get_model_lock",
                return_value=RecordingLock(),
            ),
            pytest.raises(ExceptionGroup) as raised,
        ):
            _run(
                write_reconciled_data(
                    np.array([[1.0]]),
                    ["input"],
                    np.array([[2.0]]),
                    ["output"],
                    model_id="test-model",
                    tags=["TRAINING"],
                    id_="lock-failure-1",
                )
            )

        assert any(
            isinstance(error, RuntimeError) and str(error) == "input write failed"
            for error in raised.value.exceptions
        )
        assert events[-1] == "exit"
        assert all(event.startswith("finished:") for event in events[1:-1])

    def test_deletes_partial_payloads_after_write(self) -> None:
        """Partial payloads are cleaned up after successful write."""
        input_arr = np.array([[1.0]])
        output_arr = np.array([[2.0]])

        _run(
            write_reconciled_data(
                input_arr,
                ["input"],
                output_arr,
                ["output"],
                model_id="m",
                tags=["t"],
                id_="cleanup-1",
            )
        )

        assert (
            self.mock_storage.delete_partial_payload.call_count
            == _EXPECTED_PARTIAL_DELETES
        )
        calls = self.mock_storage.delete_partial_payload.call_args_list
        ids_and_flags = [(c[0][0], c[1]["is_input"]) for c in calls]
        assert ("cleanup-1", True) in ids_and_flags
        assert ("cleanup-1", False) in ids_and_flags

    def test_adds_model_to_known(self) -> None:
        """write_reconciled_data registers the model as known."""
        input_arr = np.array([[1.0]])
        output_arr = np.array([[2.0]])

        _run(
            write_reconciled_data(
                input_arr,
                ["input"],
                output_arr,
                ["output"],
                model_id="new-model",
                tags=["t"],
                id_="known-1",
            )
        )

        self.mock_data_source.add_model_to_known.assert_called_once_with("new-model")

    def test_metadata_update_failure_is_non_fatal(self) -> None:
        """If get_metadata raises, write_reconciled_data logs but does not fail."""
        self.mock_data_source.get_metadata = mock.AsyncMock(
            side_effect=RuntimeError("metadata unavailable")
        )
        input_arr = np.array([[1.0]])
        output_arr = np.array([[2.0]])

        # Should not raise
        _run(
            write_reconciled_data(
                input_arr,
                ["input"],
                output_arr,
                ["output"],
                model_id="m",
                tags=["t"],
                id_="meta-fail",
            )
        )

        # Partial payloads should still be cleaned up
        assert (
            self.mock_storage.delete_partial_payload.call_count
            == _EXPECTED_PARTIAL_DELETES
        )

    def test_metadata_contains_tags(self) -> None:
        """Metadata array includes the tags in the correct column."""
        input_arr = np.array([[1.0], [2.0]])
        output_arr = np.array([[10.0], [20.0]])

        _run(
            write_reconciled_data(
                input_arr,
                ["input"],
                output_arr,
                ["output"],
                model_id="tag-model",
                tags=["MY_TAG"],
                id_="tag-1",
            )
        )

        # Find the metadata write call
        for call in self.mock_storage.write_data.call_args_list:
            if call[0][0] == "tag-model_metadata":
                metadata_arr = call[0][1]
                metadata_names = call[0][2]
                break
        else:
            self.fail("Metadata write call not found")

        assert "tags" in metadata_names
        tags_idx = metadata_names.index("tags")
        assert metadata_arr[0, tags_idx] == ["MY_TAG"]
        assert metadata_arr[1, tags_idx] == ["MY_TAG"]

    def test_metadata_ids_are_sequential(self) -> None:
        """Metadata IDs are formatted as {id_}_{row_index}."""
        input_arr = np.array([[1.0], [2.0], [3.0]])
        output_arr = np.array([[10.0], [20.0], [30.0]])

        _run(
            write_reconciled_data(
                input_arr,
                ["input"],
                output_arr,
                ["output"],
                model_id="id-model",
                tags=["t"],
                id_="seq",
            )
        )

        for call in self.mock_storage.write_data.call_args_list:
            if call[0][0] == "id-model_metadata":
                metadata_arr = call[0][1]
                metadata_names = call[0][2]
                break
        else:
            self.fail("Metadata write call not found")

        id_idx = metadata_names.index("id")
        assert metadata_arr[0, id_idx] == "seq_0"
        assert metadata_arr[1, id_idx] == "seq_1"
        assert metadata_arr[2, id_idx] == "seq_2"


class TestValidatePayloadType(unittest.TestCase):
    """Test the _validate_payload_type helper."""

    def test_matching_type_passes(self) -> None:
        """No exception when type matches."""
        _validate_payload_type("hello", str)  # should not raise

    def test_wrong_type_raises_http_500(self) -> None:
        """HTTPException with 500 when type does not match."""
        with pytest.raises(HTTPException) as exc_info:
            _validate_payload_type(42, str)
        assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert "invalid payload type" in exc_info.value.detail.lower()


if __name__ == "__main__":
    unittest.main()
