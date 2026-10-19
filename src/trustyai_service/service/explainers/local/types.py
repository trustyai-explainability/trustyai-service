"""Protocol-neutral types shared by local explainers and model providers."""

from enum import StrEnum


class TaskType(StrEnum):
    """Semantic task type of the explained model."""

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
