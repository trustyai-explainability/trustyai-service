from typing import Callable

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix


def filter_rows_by_inputs(data: np.ndarray, filter_func: Callable[[np.ndarray], bool]) -> np.ndarray:
    return data[np.apply_along_axis(filter_func, 1, data)]


def calculate_confusion_matrix(test: np.ndarray, truth: np.ndarray, positive_class: int) -> dict[str, int]:
    # cast test and truth to int
    test = test.astype(int)
    truth = truth.astype(int)
    # calculate confusion matrix
    cm = confusion_matrix(truth, test, labels=[positive_class, 1 - positive_class])
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def validate_fairness_groups(privileged: np.ndarray, unprivileged: np.ndarray) -> None:
    empty_groups = []
    if len(privileged) == 0:
        empty_groups.append("privileged")
    if len(unprivileged) == 0:
        empty_groups.append("unprivileged")

    if empty_groups:
        raise ValueError(f"Arrays cannot be empty for the following groups: {', '.join(empty_groups)}")


def calculate_favorable_probability(group: np.ndarray, favorable_outputs: np.ndarray) -> float:
    """
    Calculate the probability of favorable outcomes in a group.

    :param group: NumPy array with the group data (outcomes in last column)
    :param favorable_outputs: NumPy array of outputs considered favorable/desirable
    :return: Probability of favorable outcomes in the group
    """
    return np.sum(np.isin(group[:, -1], favorable_outputs)) / len(group)


def split_by_privilege(
    samples: np.ndarray,
    model: ClassifierMixin,
    privilege_columns: list[int],
    privilege_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split samples into privileged and unprivileged groups based on model predictions.

    :param samples: NumPy array of inputs to be used for testing fairness
    :param model: The model to be tested for fairness
    :param privilege_columns: List of integers specifying the indices of the privileged columns
    :param privilege_values: NumPy array of integers specifying the privileged values
    :return: Tuple of (privileged, unprivileged) groups with predictions appended
    """
    outputs = model.predict(samples)
    data = np.append(samples, outputs.reshape(-1, 1), axis=1)
    privileged = data[np.all(data[:, privilege_columns] == privilege_values, axis=1)]
    unprivileged = data[np.all(data[:, privilege_columns] != privilege_values, axis=1)]
    return privileged, unprivileged
