"""Pure LIME callable-core tests."""

import numpy as np

from trustyai_service.core.explainers.local.lime import compute_lime_explanation


class _Explanation:
    mode = "classification"

    def __init__(self) -> None:
        self.score = {0: 0.1, 1: 0.9}
        self.local_pred = {0: [0.2], 1: [0.8]}
        self.intercept = {0: 0.05, 1: 0.15}

    def available_labels(self) -> list[int]:
        return [0, 1]

    def as_list(self, *, label: int) -> list[tuple[str, float]]:
        return [(f"f{label}", float(label))]


class _Explainer:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def explain_instance(
        self, instance: np.ndarray, predict_fn: object, **kwargs: object
    ) -> _Explanation:
        del instance, predict_fn, kwargs
        self.calls.append(1)
        return _Explanation()


def test_lime_core_preserves_explicit_class_selection() -> None:
    result = compute_lime_explanation(
        _Explainer(),
        np.array([1.0, 2.0]),
        lambda values: np.column_stack((values[:, 0], 1 - values[:, 0])),
        num_samples=4,
        num_features=2,
        label=1,
    )
    assert result == ([("f1", 1.0)], 0.9, 0.8, 0.15)
