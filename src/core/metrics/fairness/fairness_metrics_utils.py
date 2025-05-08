import numpy as np

def filter_rows_by_inputs(data, filter_func):
    return data[np.apply_along_axis(filter_func, 1, data)]

def calculate_confusion_matrix(test: np.array, truth: np.array, positive_class: int) -> dict:
    tp = np.sum((test == positive_class) & (truth == positive_class))
    tn = np.sum((test != positive_class) & (truth != positive_class))
    fp = np.sum((test == positive_class) & (truth != positive_class))
    fn = np.sum((test != positive_class) & (truth == positive_class))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}