"""Contract constraints for the ModelMesh partial inference payload.

consume_inference_payload() answers HTTP 400 when id, kind, modelid or data is
falsy. The schema must declare those fields required without moving the
rejection into Pydantic validation (which would answer 422 instead).
"""

from trustyai_service.endpoints.consumer import InferencePartialPayload


def test_declares_the_fields_the_handler_requires() -> None:
    """A payload of {} is rejected with 400, so those fields are required."""
    schema = InferencePartialPayload.model_json_schema()

    assert set(schema["required"]) == {"id", "kind", "modelid", "data"}
    assert "metadata" not in schema["required"]


def test_runtime_stays_permissive() -> None:
    """The constraint is declared in the schema only; HTTP 400 is unchanged."""
    payload = InferencePartialPayload()

    assert payload.id is None
    assert payload.metadata == {}
