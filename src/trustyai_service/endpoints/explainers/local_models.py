"""Shared request model for real-model and explicit-surrogate explainers."""

from __future__ import annotations

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator

from trustyai_service.service.explainers.local.types import PredictionSource, TaskType

_CONTROL_CHAR_LIMIT = 32


class LocalExplanationModelConfig(BaseModel):
    """KServe model identity and prediction-source contract."""

    base_url: AnyHttpUrl | None = None
    model_name: str = Field(min_length=1)
    model_version: str | None = None
    prediction_source: PredictionSource = PredictionSource.MODEL
    task: TaskType
    input_name: str | None = None
    output_name: str | None = None

    @field_validator("model_name", "model_version", "input_name", "output_name")
    @classmethod
    def validate_segment(cls, value: str | None) -> str | None:
        """Reject tensor and model names that could escape the URL path."""
        if value is not None and (
            not value.strip()
            or value in {".", ".."}
            or any(c in value for c in "/\\%")
            or any(ord(c) < _CONTROL_CHAR_LIMIT for c in value)
        ):
            msg = "model and tensor names must be non-blank path-safe segments"
            raise ValueError(msg)
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: AnyHttpUrl | None) -> AnyHttpUrl | None:
        """Require a credential-free HTTP(S) model endpoint."""
        if value is not None and (
            value.scheme not in {"http", "https"}
            or value.username is not None
            or value.password is not None
        ):
            msg = "base_url must be an HTTP(S) URL without credentials"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_source(self) -> LocalExplanationModelConfig:
        """Require a model endpoint when predictions come from a real model."""
        if self.prediction_source is PredictionSource.MODEL and self.base_url is None:
            msg = "base_url is required when prediction_source=MODEL"
            raise ValueError(msg)
        return self
