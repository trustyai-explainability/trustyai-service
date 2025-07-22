# pylint: disable=line-too-long
from typing import List

import numpy as np
from sklearn.base import ClassifierMixin


class GroupStatisticalParityDifference:
    """
    Calculate group statistical parity difference (SPD).
    """

    @staticmethod
    def calculate_model(
        samples: np.ndarray,
        model: ClassifierMixin,
        privilege_columns: List[int],
        privilege_values: List[int],
        favorable_output,
    ) -> float:
        """
        Calculate group statistical parity difference (SPD) for model outputs.
        :param samples a NumPy array of inputs to be used for testing fairness
        :param model the model to be tested for fairness
        :param privilege_columns a list of integers specifying the indices of the privileged columns
        :param privilege_values a list integers specifying the privileged values
        :param favorable_output the outputs that are considered favorable / desirable
        return SPD score
        """
        outputs = model.predict(samples)
        data = np.append(samples, outputs, axis=1)
        privileged = np.all(data[:, privilege_columns] == privilege_values, axis=1)
        unprivileged = np.all(data[:, privilege_columns] != privilege_values, axis=1)

        return GroupStatisticalParityDifference.calculate(privileged, unprivileged, favorable_output)

    @staticmethod
    def calculate(
        privileged,
        unprivileged,
        favorable_output,
    ) -> float:
        """
        Calculate statistical/demographic parity difference (SPD) when the labels are pre-calculated.
        :param priviledged numPy array with the privileged groups
        :param unpriviledged numPy array with the unpriviledged groups
        :param favorableOutput an output that is considered favorable / desirable
        return SPD, between -1 and 1
        """
        probability_privileged = np.sum(privileged[:, -1] == favorable_output) / len(privileged)
        probability_unprivileged = np.sum(unprivileged[:, -1] == favorable_output) / len(unprivileged)
        return probability_unprivileged - probability_privileged
