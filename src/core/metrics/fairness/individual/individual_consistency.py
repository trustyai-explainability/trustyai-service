"""Individual fairness consistency metric implementation."""

from collections.abc import Callable

import numpy as np
from sklearn.base import ClassifierMixin


class IndividualConsistency:
    """Calculate individual fairness in terms of consistency of predictions across similar inputs."""

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
        :param model: the model under inspection
        :return: the consistency measure in [0, 1]
        """
        if len(samples) == 0:
            msg = "Samples cannot be empty - cannot compute consistency."
            raise ValueError(msg)
        mismatches = 0
        total_comparisons = 0
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
                output_arr = np.asarray(output)
                for neighbor_output in neighbors_outputs:
                    neighbor_arr = np.asarray(neighbor_output)
                    if neighbor_arr.shape != output_arr.shape:
                        msg = (
                            "Prediction outputs have inconsistent shapes - "
                            "cannot compute consistency."
                        )
                        raise ValueError(msg)
                    mismatches += int(np.sum(neighbor_arr != output_arr))
            total_comparisons += len(neighbors) * output_width * len(prediction_outputs)
        return 1.0 - (mismatches / total_comparisons) if total_comparisons > 0 else 1.0
