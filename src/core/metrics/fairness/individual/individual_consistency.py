# pylint: disable=too-few-public-methods, line-too-long
"""Individual fairness consistency metric implementation."""

from collections.abc import Callable

import numpy as np
from sklearn.base import ClassifierMixin


class IndividualConsistency:
    """Calculate individual fairness in terms of consistency of predictions across similar inputs.

    :param proximity_function: a function that finds the top k similar
        inputs, given a reference input and a list of inputs :param
        samples a list of inputs to be tested for consistency :param
        predictionProvider the model under inspection return the
        consistency measure
    """

    @staticmethod
    def calculate(
        proximity_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        samples: np.ndarray,
        model: ClassifierMixin,
    ) -> float:
        """Calculate individual fairness.

        :param proximity_function: a function that finds the top k
            similar inputs, given a reference input and a list of inputs
        :param samples: a list of inputs to be tested for consistency
        :param model: the model under inspection return the consistency
            measure
        """
        consistency = 1.0
        for sample in samples:
            prediction_outputs = model.predict(sample)
            if len(prediction_outputs) == 0:
                msg = "Model output cannot be empty."
                raise ValueError(msg)
            prediction_output = prediction_outputs[0]
            # Handle both scalar and array predictions
            output_width = int(np.size(prediction_output))

            # Validate non-zero divisor terms to prevent ZeroDivisionError
            if output_width == 0:
                msg = "Model output has zero size - cannot compute consistency."
                raise ValueError(msg)

            neighbors = proximity_function(sample, samples)

            if len(neighbors) == 0:
                msg = "Proximity function returned no neighbors - cannot compute consistency."
                raise ValueError(msg)

            neighbors_outputs = model.predict(neighbors)
            for output in prediction_outputs:
                for neighbor_output in neighbors_outputs:
                    # Use array-safe comparison (NumPy 2.0+ requires this for arrays)
                    if not np.array_equal(neighbor_output, output):
                        consistency -= 1 / (
                            len(neighbors) * output_width * len(samples)
                        )
        return consistency
