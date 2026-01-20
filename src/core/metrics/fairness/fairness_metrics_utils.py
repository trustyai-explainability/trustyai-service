from typing import Callable
import numpy as np
from sklearn.metrics import confusion_matrix


def filter_rows_by_inputs(data: np.ndarray, filter_func: Callable[[np.ndarray], bool]):
    return data[np.apply_along_axis(filter_func, 1, data)]


def calculate_confusion_matrix(test: np.array, truth: np.array, positive_class: int) -> dict:
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
