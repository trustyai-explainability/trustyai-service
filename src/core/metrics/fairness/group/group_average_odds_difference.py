# pylint: disable=line-too-long, too-many-arguments
from typing import List, Any

import numpy as np

from src.core.metrics.fairness.fairness_metrics_utils import filter_rows_by_inputs, calculate_confusion_matrix

class GroupAverageOddsDifference:
    """
    Calculate group average odds difference.
    """
    @ staticmethod
    def calculate_model(
        samples: np.ndarray,
        model: Any,
        privilege_columns: List[int],
        privilege_values: List[int],
        postive_class: List[int],
        output_column: int
    ):
        """
        Calculate group average odds difference for model outputs.
        :param samples a NumPy arrary of inputs to be used for testing fairness
        :param model the model to be tested for fairness
        :param privilege_columns a list of integers specifying the indices of the privileged columns
        :param privilege_values a list of intergers specifying the privileged values
        :param postive_class the favorable / desirable outputs
        :param output_column the column index where the output is located
        return group average odds difference score
        """
        outputs = model.predict(samples)
        truth = np.append(samples, outputs, axis=1)

        return GroupAverageOddsDifference.calculate(samples, truth, privilege_columns, privilege_values, postive_class, output_column)

    @staticmethod
    def calculate(test, truth, privilege_columns, privilege_values, positive_class, output_column):
        """
        Calculate group average odds difference when the labels are pre-calculated.
        :param test a NumPy array representing the test data
        :param truth a NumPy array representing the truth data
        :param privilege_columns a list of integers specifying the indices of the privileged columns
        :param privilege_values a list of intergers specifying the privileged values
        :param positive_class the favorable / desirable outputs
        :param output_column the column where the output is located
        return group average odds difference, between -1 and 1
        """
        def privilege_filter(row):
            return np.array_equal(row[privilege_columns], privilege_values)

        test_privileged = filter_rows_by_inputs(test, privilege_filter)
        test_unprivileged = filter_rows_by_inputs(test, lambda row: not privilege_filter(row))

        truth_privileged = filter_rows_by_inputs(truth, privilege_filter)
        truth_unprivileged = filter_rows_by_inputs(truth, lambda row: not privilege_filter(row))

        ucm = calculate_confusion_matrix(test_unprivileged[:, output_column], truth_unprivileged[:, output_column], positive_class)
        pcm = calculate_confusion_matrix(test_privileged[:, output_column], truth_privileged[:, output_column], positive_class)

        utp, utn, ufp, ufn = ucm["tp"], ucm["tn"], ucm["fp"], ucm["fn"]
        ptp, ptn, pfp, pfn = pcm["tp"], pcm["tn"], pcm["fp"], pcm["fn"]

        return (utp / (utp + ufn + 1e-10) - ptp / (ptp + pfn + 1e-10)) / 2 + \
            (ufp / (ufp + utn + 1e-10) - pfp / (pfp + ptn + 1e-10)) / 2
