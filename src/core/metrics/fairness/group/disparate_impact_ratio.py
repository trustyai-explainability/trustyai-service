# pylint: disable=line-too-long
from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin

from src.core.metrics.fairness.fairness_metrics_utils import validate_fairness_groups


class DisparateImpactRatio:
    """
    Calculate disparate impact ratio (DIR).
    """

    @staticmethod
    def calculate_model(
        samples: np.ndarray,
        model: ClassifierMixin,
        privilege_columns: List[int],
        privilege_values: List[int],
        favorable_output: np.ndarray,
    ) -> float:
        """
        Calculate disparate impact ratio (DIR) for model outputs.
        :param samples a NumPy array of inputs to be used for testing fairness
        :param model the model to be tested for fairness
        :param privilege_columns a list of integers specifying the indices of the privileged columns
        :param privilege_values a list integers specifying the privileged values
        :param favorable_output the outputs that are considered favorable / desirable
        return DIR score
        """
        outputs = model.predict(samples)
        data = np.append(samples, outputs, axis=1)
        privileged = np.sum(data[:, privilege_columns] == privilege_values)
        unprivileged = np.sum(data[:, privilege_columns] != privilege_values)

        return DisparateImpactRatio.calculate(privileged, unprivileged, favorable_output)

    @staticmethod
    def calculate(
        privileged: Union[int, np.ndarray], unprivileged: Union[int, np.ndarray], favorable_output: int
    ) -> float:
        """
        Calculate disparate impact ratio (DIR) when the labels are pre-calculated.
        :param privileged a NumPy array with the privileged groups
        :param unprivileged a NumPy array with the unprivileged groups
        :param favorableOutput an output that is considered favorable / desirable
        return DIR, between 0 and 1
        """
        validate_fairness_groups(privileged=privileged, unprivileged=unprivileged)

        probability_privileged = np.sum(privileged[:, -1] == favorable_output) / len(privileged)
        probability_unprivileged = np.sum(unprivileged[:, -1] == favorable_output) / len(unprivileged)
        return probability_unprivileged / probability_privileged
