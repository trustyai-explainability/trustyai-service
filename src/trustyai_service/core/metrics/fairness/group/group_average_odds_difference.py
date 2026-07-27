# pylint: disable=line-too-long, too-many-arguments
"""Group average odds difference fairness metric implementation."""

import numpy as np
from sklearn.base import ClassifierMixin

from trustyai_service.core.metrics.fairness.fairness_metrics_utils import (
    calculate_confusion_matrix,
    filter_rows_by_inputs,
)


class GroupAverageOddsDifference:
    """Calculate group average odds difference."""

    @staticmethod
    def calculate_model(
        samples: np.ndarray,
        model: ClassifierMixin,
        privilege_columns: list[int],
        privilege_values: list[int],
        postive_class: int,
        output_column: int,
    ) -> float:
        """Calculate group average odds difference for model outputs.

        :param samples: NumPy array of inputs for testing fairness.
        :param model: Model to be tested for fairness.
        :param privilege_columns: Indices of the privileged columns.
        :param privilege_values: Privileged values.
        :param postive_class: Favorable/desirable outputs.
        :param output_column: Column index where the output is located.
        :return: Group average odds difference score.
        """
        outputs = model.predict(samples)
        truth = np.append(samples, outputs, axis=1)

        return GroupAverageOddsDifference.calculate(
            samples,
            truth,
            privilege_columns,
            privilege_values,
            postive_class,
            output_column,
        )

    @staticmethod
    def calculate(
        test: np.ndarray,
        truth: np.ndarray,
        privilege_columns: list[int],
        privilege_values: list[int],
        positive_class: int,
        output_column: int,
    ) -> float:
        """Calculate group average odds difference when the labels are pre-calculated.

        :param test: NumPy array representing the test data.
        :param truth: NumPy array representing the truth data.
        :param privilege_columns: Indices of the privileged columns.
        :param privilege_values: Privileged values.
        :param positive_class: Favorable/desirable outputs.
        :param output_column: Column where the output is located.
        :return: Group average odds difference, between -1 and 1.
        """

        def privilege_filter(row: np.ndarray) -> bool:
            return bool(np.array_equal(row[privilege_columns], privilege_values))

        test_privileged = filter_rows_by_inputs(test, privilege_filter)
        test_unprivileged = filter_rows_by_inputs(
            test,
            lambda row: not privilege_filter(row),
        )

        truth_privileged = filter_rows_by_inputs(truth, privilege_filter)
        truth_unprivileged = filter_rows_by_inputs(
            truth,
            lambda row: not privilege_filter(row),
        )

        ucm = calculate_confusion_matrix(
            test_unprivileged[:, output_column],
            truth_unprivileged[:, output_column],
            positive_class,
        )
        pcm = calculate_confusion_matrix(
            test_privileged[:, output_column],
            truth_privileged[:, output_column],
            positive_class,
        )

        utp, utn, ufp, ufn = ucm["tp"], ucm["tn"], ucm["fp"], ucm["fn"]
        ptp, ptn, pfp, pfn = pcm["tp"], pcm["tn"], pcm["fp"], pcm["fn"]

        return (utp / (utp + ufn + 1e-10) - ptp / (ptp + pfn + 1e-10)) / 2 + (
            ufp / (ufp + utn + 1e-10) - pfp / (pfp + ptn + 1e-10)
        ) / 2
