"""Small enums shared by local explainer request and service layers."""

from enum import StrEnum


class PredictionSource(StrEnum):
    """Source used to produce predictions for an explanation."""

    MODEL = "MODEL"
    SURROGATE = "SURROGATE"


class TaskType(StrEnum):
    """Semantic task of the explained model."""

    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
