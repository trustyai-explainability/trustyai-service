# pylint: disable=line-too-long
"""Group statistical parity difference (SPD) fairness metric implementation."""

import numpy as np
from sklearn.base import ClassifierMixin

from trustyai_service.core.metrics.fairness.fairness_metrics_utils import (
    calculate_favorable_probability,
    split_by_privilege,
    validate_fairness_groups,
)


class GroupStatisticalParityDifference:
    """Calculate group statistical parity difference (SPD)."""

    @staticmethod
    def calculate_model(
        samples: np.ndarray,
        model: ClassifierMixin,
        privilege_columns: list[int],
        privilege_values: np.ndarray,
        favorable_outputs: np.ndarray,
    ) -> float:
        """Calculate group statistical parity difference (SPD) for model outputs.

        :param samples: NumPy array of inputs for testing fairness.
        :param model: Model to be tested for fairness.
        :param privilege_columns: Indices of the privileged columns.
        :param privilege_values: Privileged values.
        :param favorable_outputs: Outputs considered favorable/desirable.
        :return: SPD score.
        """
        privileged, unprivileged = split_by_privilege(
            samples,
            model,
            privilege_columns,
            privilege_values,
        )
        return GroupStatisticalParityDifference.calculate(
            privileged,
            unprivileged,
            favorable_outputs,
        )

    @staticmethod
    def calculate(
        privileged: np.ndarray,
        unprivileged: np.ndarray,
        favorable_outputs: np.ndarray,
    ) -> float:
        """Calculate statistical/demographic parity difference (SPD) when labels are pre-calculated.

        :param privileged: NumPy array with the privileged groups.
        :param unprivileged: NumPy array with the unprivileged groups.
        :param favorable_outputs: Output considered favorable/desirable.
        :return: SPD, between -1 and 1.
        """
        validate_fairness_groups(privileged=privileged, unprivileged=unprivileged)

        probability_privileged = calculate_favorable_probability(
            privileged,
            favorable_outputs,
        )
        probability_unprivileged = calculate_favorable_probability(
            unprivileged,
            favorable_outputs,
        )
        return probability_unprivileged - probability_privileged
