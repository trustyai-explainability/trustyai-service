"""Contract-only constraints for drift request bodies.

The handlers reject a missing ``referenceTag`` and an explicit empty
``fitColumns`` list with HTTP 400 (``validate_drift_request``). Declaring those
constraints on the Pydantic fields would make FastAPI answer 422 first and
change the API, so this hook rewrites only the JSON schema: the contract states
what the handlers require while requests keep their runtime behavior.
"""

from typing import Any

from pydantic.json_schema import GetJsonSchemaHandler, JsonSchemaValue
from pydantic_core import CoreSchema


def require_drift_fields(
    core_schema: CoreSchema,
    handler: GetJsonSchemaHandler,
) -> JsonSchemaValue:
    """Declare ``referenceTag`` required and a non-empty ``fitColumns`` array.

    ``fitColumns`` stays optional (omitted or ``null`` still derives the
    columns from metadata); only the explicit array branch gains
    ``minItems: 1``.
    """
    json_schema: dict[str, Any] = handler(core_schema)

    if "referenceTag" in json_schema.get("properties", {}):
        required = json_schema.setdefault("required", [])
        if "referenceTag" not in required:
            required.append("referenceTag")
        # A required field with a default suggests it may be omitted.
        json_schema["properties"]["referenceTag"].pop("default", None)

    fit_columns = json_schema.get("properties", {}).get("fitColumns", {})
    for branch in fit_columns.get("anyOf", []):
        if branch.get("type") == "array":
            branch["minItems"] = 1

    return json_schema


class DriftRequestContract:
    """Mixin adding the drift request contract constraints to a request model."""

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Apply the contract constraints to the generated model schema."""
        return require_drift_fields(core_schema, handler)
