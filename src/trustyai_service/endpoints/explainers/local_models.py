"""Canonical request models shared by local LIME and SHAP explainers."""

from __future__ import annotations

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, field_validator

from trustyai_service.service.explainers.local.model_provider import (
    ProviderInvalidRequestError,
)
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType

_CONTROL_CHAR_LIMIT = 32
_DEL_CHAR = 127


def _contains_control_character(value: str) -> bool:
    """Return whether a value contains an HTTP control character."""
    return any(
        ord(character) < _CONTROL_CHAR_LIMIT or ord(character) == _DEL_CHAR
        for character in value
    )


class LocalExplanationModelConfig(BaseModel):
    """KServe model identity and prediction-source contract."""

    model_config = ConfigDict(extra="forbid")

    base_url: AnyHttpUrl | None = None
    model_name: str
    model_version: str | None = None
    prediction_source: PredictionSource = PredictionSource.MODEL
    task: TaskType
    input_name: str | None = None
    output_name: str | None = None

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
    def validate_base_url(cls, value: AnyHttpUrl | None) -> AnyHttpUrl | None:
        """Reject credentials and unsafe components in the KServe base URL."""
        if value is None:
            return None

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


def validate_local_explanation_model_config(
    config: LocalExplanationModelConfig,
) -> LocalExplanationModelConfig:
    """Validate source semantics after Pydantic has parsed the request."""
    if config.prediction_source is PredictionSource.MODEL and config.base_url is None:
        msg = "base_url is required when prediction_source=MODEL"
        raise ProviderInvalidRequestError(msg)
    return config


__all__ = [
    "LocalExplanationModelConfig",
    "validate_local_explanation_model_config",
]
