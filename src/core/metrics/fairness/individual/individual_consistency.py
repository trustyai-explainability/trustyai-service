# pylint: disable=too-few-public-methods, line-too-long
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin

class IndividualConsistency:
    """
    Calculate individual fairness in terms of consistency of predictions across similar inputs.
    :param proximity_function: a function that finds the top k similar inputs, given a reference input and a list of inputs
    :param samples a list of inputs to be tested for consistency
    :param predictionProvider the model under inspection
    return the consistency measure
    """
    @staticmethod
    def calculate(
        proximity_function: Any,
        samples: np.ndarray,
        model: ClassifierMixin
    ) -> float:
        """
        Calculate individual fairness.
        :param proximity_function: a function that finds the top k similar inputs, given a reference input and a list of inputs
        :param samples: a list of inputs to be tested for consistency
        :param model: the model under inspection
        return the consistency measure
        """
        consistency =  1
        for sample in samples:
            prediction_outputs = model.predict(sample)
            if len(prediction_outputs) == 0:
                raise ValueError("Model output cannot be empty.")
            prediction_output = prediction_outputs[0]
            neighbors = proximity_function(sample, samples)
            neighbors_outputs = model.predict(neighbors)
            for output in prediction_outputs:
                for neighbor_output in neighbors_outputs:
                    if neighbor_output != output:
                        consistency -= 1 / (len(neighbors) * len(prediction_output) * len(samples))
        return consistency
