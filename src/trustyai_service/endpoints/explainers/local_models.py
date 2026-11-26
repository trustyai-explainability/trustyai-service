"""Canonical request models shared by local LIME and SHAP explainers."""

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, field_validator

from trustyai_service.service.explainers.local.model_provider import (
    NUMERIC_TENSOR_DATATYPES,
)
from trustyai_service.service.explainers.local.types import TaskType

_CONTROL_CHAR_LIMIT = 32
_DEL_CHAR = 127


def _contains_control_character(value: str) -> bool:
    """Return whether a value contains an HTTP control character."""
    return any(
        ord(character) < _CONTROL_CHAR_LIMIT or ord(character) == _DEL_CHAR
        for character in value
    )


class LocalExplanationModelConfig(BaseModel):
    """KServe model identity and task contract."""

    model_config = ConfigDict(extra="forbid")

    base_url: AnyHttpUrl
    model_name: str
    model_version: str | None = None
    task: TaskType
    input_datatype: str = "FP64"
    input_name: str | None = None
    output_name: str | None = None

    @field_validator("input_datatype")
    @classmethod
    def validate_input_datatype(cls, value: str) -> str:
        """Restrict the request-supplied datatype to numeric tensors."""
        if value not in NUMERIC_TENSOR_DATATYPES:
            msg = "input_datatype must be a supported numeric tensor datatype"
            raise ValueError(msg)
        return value

    @field_validator("model_name", "model_version", "input_name", "output_name")
    @classmethod
    def validate_path_segment(cls, value: str | None) -> str | None:
        """Reject blank or unsafe model and tensor path segments."""
        if value is not None and (
            not value.strip()
            or value in {".", ".."}
            or any(character in value for character in "/\\%")
            or _contains_control_character(value)
        ):
            msg = "model and tensor names must be non-blank path-safe segments"
            raise ValueError(msg)
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        """Reject credentials and unsafe components in the KServe base URL."""
        path_segments = value.path.split("/")
        if (
            value.username is not None
            or value.password is not None
            or value.query is not None
            or value.fragment is not None
            or "%" in value.path
            or any(segment in {".", ".."} for segment in path_segments)
            or _contains_control_character(value.path)
        ):
            msg = "base_url must be an HTTP(S) URL without credentials or unsafe components"
            raise ValueError(msg)
        return value


__all__ = ["LocalExplanationModelConfig"]
