"""Protocol-neutral types shared by local explainers and model providers."""

from enum import StrEnum


class PredictionSource(StrEnum):
    """Source used to produce predictions for an explanation."""

    MODEL = "MODEL"
    SURROGATE = "SURROGATE"


class TaskType(StrEnum):
    """Semantic task type of the explained model."""

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
