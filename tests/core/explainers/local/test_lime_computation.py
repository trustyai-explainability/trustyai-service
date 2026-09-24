"""Tests for the pure callable-based LIME core."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import trustyai_service.core.explainers.local.lime as lime_core
from trustyai_service.core.explainers.local.lime import (
    LIMEConfidenceIntervalResult,
    LIMEExplanationResult,
    compute_lime_confidence_intervals,
    compute_lime_explanation,
    create_lime_explainer,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small deterministic background and explained instance."""
    data = np.asarray(
        [
            [-2.0, 0.5, 1.0],
            [-1.0, 1.0, 0.0],
            [0.0, -0.5, 2.0],
            [1.0, 1.5, -1.0],
            [2.0, -1.0, 0.5],
            [3.0, 0.0, -2.0],
            [4.0, 2.0, 1.5],
            [5.0, -1.5, -0.5],
        ]
    )
    return data, np.asarray([1.25, 0.25, 0.75])


def _regression_predict(values: np.ndarray) -> np.ndarray:
    """Return a deterministic scalar output for every row."""
    return 2.0 * values[:, 0] - values[:, 1] + 0.5 * values[:, 2]


def _classification_predict(values: np.ndarray) -> np.ndarray:
    """Return deterministic binary probabilities for every row."""
    score = 0.25 * values[:, 0] - 0.5 * values[:, 1] + values[:, 2]
    positive = 1.0 / (1.0 + np.exp(-score))
    return np.column_stack((1.0 - positive, positive))


def _make_regression_explainer(seed: int = 17) -> object:
    """Create a real LIME regression explainer for core tests."""
    training_data, _ = _regression_data()
    return create_lime_explainer(
        training_data,
        ["f0", "f1", "f2"],
        classification=False,
        seed=seed,
    )


def _make_classification_explainer(seed: int = 17) -> object:
    """Create a real LIME classification explainer for core tests."""
    training_data, _ = _regression_data()
    return create_lime_explainer(
        training_data,
        ["f0", "f1", "f2"],
        classification=True,
        seed=seed,
    )


def test_core_signatures_make_algorithm_options_keyword_only() -> None:
    """Expose only the planned positional inputs and keyword-only options."""
    create_signature = inspect.signature(create_lime_explainer)
    explanation_signature = inspect.signature(compute_lime_explanation)
    confidence_signature = inspect.signature(compute_lime_confidence_intervals)

    assert [
        parameter.name
        for parameter in create_signature.parameters.values()
        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD
    ] == ["training_data", "feature_names"]
    assert (
        create_signature.parameters["classification"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    for name in ("num_samples", "num_features", "label"):
        assert (
            explanation_signature.parameters[name].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
    for name in (
        "num_samples",
        "num_features",
        "confidence",
        "n_bootstrap",
        "seed",
        "label",
    ):
        assert (
            confidence_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        )


def test_importing_core_does_not_load_optional_or_service_integrations() -> None:
    """Import the core without importing LIME, HTTP, FastAPI, or storage code."""
    repository_root = Path(__file__).parents[4]
    script = """
import builtins
import sys

sys.path.insert(0, "src")
forbidden = (
    "lime",
    "httpx",
    "fastapi",
    "trustyai_service.service",
    "trustyai_service.endpoints",
)
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == module or name.startswith(module + ".") for module in forbidden):
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import trustyai_service.core.explainers.local.lime
"""

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not lime_core._LIME_AVAILABLE, reason="lime is not installed")
def test_create_explainer_maps_boolean_task_and_preserves_feature_names() -> None:
    """Use the installed LIME constructor with the explicit boolean task."""
    training_data, _ = _regression_data()

    regression = create_lime_explainer(
        training_data,
        ("f0", "f1", "f2"),
        classification=False,
        seed=11,
    )
    classification = create_lime_explainer(
        training_data,
        ("f0", "f1", "f2"),
        classification=True,
        seed=11,
    )

    assert regression.mode == "regression"
    assert classification.mode == "classification"
    assert regression.feature_names == ["f0", "f1", "f2"]
    assert classification.feature_names == ["f0", "f1", "f2"]
    assert regression.discretizer is None


@pytest.mark.skipif(not lime_core._LIME_AVAILABLE, reason="lime is not installed")
def test_regression_explanation_is_deterministic_and_respects_feature_limit() -> None:
    """Repeat a seeded regression explanation and return at most requested features."""
    _, instance = _regression_data()
    first = compute_lime_explanation(
        _make_regression_explainer(),
        instance,
        _regression_predict,
        num_samples=80,
        num_features=2,
        label=None,
    )
    second = compute_lime_explanation(
        _make_regression_explainer(),
        instance,
        _regression_predict,
        num_samples=80,
        num_features=2,
        label=None,
    )

    assert isinstance(first, LIMEExplanationResult)
    assert first == second
    assert len(first.feature_weights) <= 2
    assert isinstance(first.score, float)
    assert isinstance(first.local_prediction, float)
    assert isinstance(first.intercept, float)
    assert first == (
        first.feature_weights,
        first.score,
        first.local_prediction,
        first.intercept,
    )


@pytest.mark.skipif(not lime_core._LIME_AVAILABLE, reason="lime is not installed")
def test_classification_explanation_uses_the_requested_class_label() -> None:
    """Forward an explicit class label to LIME and select matching scalar fields."""
    _, instance = _regression_data()
    result = compute_lime_explanation(
        _make_classification_explainer(),
        instance,
        _classification_predict,
        num_samples=80,
        num_features=3,
        label=0,
    )

    assert isinstance(result, LIMEExplanationResult)
    assert len(result.feature_weights) <= 3
    assert np.isfinite(result.local_prediction)
    assert np.isfinite(result.intercept)


@pytest.mark.skipif(not lime_core._LIME_AVAILABLE, reason="lime is not installed")
def test_confidence_intervals_are_deterministic_for_a_seed() -> None:
    """Use the same seeded bootstrap inputs to produce identical bounds."""
    _, instance = _regression_data()
    first = compute_lime_confidence_intervals(
        _make_regression_explainer(),
        instance,
        _regression_predict,
        num_samples=60,
        num_features=2,
        confidence=0.9,
        n_bootstrap=5,
        seed=101,
        label=None,
    )
    second = compute_lime_confidence_intervals(
        _make_regression_explainer(),
        instance,
        _regression_predict,
        num_samples=60,
        num_features=2,
        confidence=0.9,
        n_bootstrap=5,
        seed=101,
        label=None,
    )

    assert isinstance(first, LIMEConfidenceIntervalResult)
    assert first == second
    assert first.lower_bounds is not None
    assert first.upper_bounds is not None
    assert first.lower_bounds.keys() == first.upper_bounds.keys()
    assert all(
        first.lower_bounds[name] <= first.upper_bounds[name]
        for name in first.lower_bounds
    )
    lower, upper = first
    assert lower == first.lower_bounds
    assert upper == first.upper_bounds


def test_confidence_one_disables_bootstrap() -> None:
    """Return an explicit disabled result without invoking the explainer."""
    result = compute_lime_confidence_intervals(
        object(),
        np.zeros(2),
        lambda values: values[:, 0],
        num_samples=10,
        num_features=1,
        confidence=1.0,
        n_bootstrap=3,
        seed=1,
        label=None,
    )

    assert result == LIMEConfidenceIntervalResult(None, None)


def test_optional_lime_dependency_failure_is_explicit_and_endpoint_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a missing optional algorithm without constructing an HTTP error."""
    monkeypatch.setattr(lime_core, "_LIME_AVAILABLE", False)

    with pytest.raises(lime_core.LIMEUnavailableError, match="LIME dependency"):
        create_lime_explainer(
            np.ones((2, 1)),
            ["f0"],
            classification=False,
        )


class _ProviderFailureError(RuntimeError):
    """Exception used to verify prediction callable identity preservation."""


@pytest.mark.skipif(not lime_core._LIME_AVAILABLE, reason="lime is not installed")
def test_prediction_callable_failure_is_not_replaced_or_wrapped() -> None:
    """A primary explanation re-raises the exact exception from predict_fn."""
    _, instance = _regression_data()
    error = _ProviderFailureError("model failed")

    def failing_predict(_: np.ndarray) -> np.ndarray:
        raise error

    with pytest.raises(_ProviderFailureError) as raised:
        compute_lime_explanation(
            _make_regression_explainer(),
            instance,
            failing_predict,
            num_samples=20,
            num_features=2,
            label=None,
        )

    assert raised.value is error


class _FakeExplanation:
    """Minimal explanation object matching the installed LIME result boundary."""

    mode = "regression"
    score = 0.5

    def __init__(self) -> None:
        self.local_pred = np.asarray([1.0])
        self.intercept = {1: 0.25}

    def available_labels(self) -> list[int]:
        """Return the regression sentinel used by LIME."""
        return [1]

    def as_list(self, *, label: int = 1) -> list[tuple[str, float]]:
        """Return one deterministic feature attribution."""
        del label
        return [("f0", 1.0)]


class _DegenerateThenWorkingExplainer:
    """Raise one numerical error, then return valid explanations."""

    feature_names = ("f0", "f1")
    mode = "regression"

    def __init__(self) -> None:
        self.calls = 0

    def explain_instance(
        self,
        data_row: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs: object,
    ) -> _FakeExplanation:
        """Record calls and exercise the prediction callable on every run."""
        del data_row, kwargs
        self.calls += 1
        predict_fn(np.zeros((1, 2)))
        if self.calls == 1:
            msg = "singular local regression"
            raise ValueError(msg)
        return _FakeExplanation()


def test_bootstrap_skips_numerical_degeneracy_but_returns_successful_bounds() -> None:
    """Skip only a numerical bootstrap failure and retain later valid samples."""
    explainer = _DegenerateThenWorkingExplainer()
    result = compute_lime_confidence_intervals(
        explainer,
        np.zeros(2),
        lambda values: values[:, 0],
        num_samples=10,
        num_features=1,
        confidence=0.9,
        n_bootstrap=3,
        seed=3,
        label=None,
    )

    assert result.lower_bounds == {"f0": 1.0, "f1": 0.0}
    assert result.upper_bounds == {"f0": 1.0, "f1": 0.0}
    assert explainer.calls == 3


def test_bootstrap_prediction_failure_preserves_identity_even_for_value_error() -> None:
    """Do not classify a prediction callable's ValueError as numerical degeneracy."""
    explainer = _DegenerateThenWorkingExplainer()
    error = ValueError("provider returned a bad response")

    def failing_predict(_: np.ndarray) -> np.ndarray:
        raise error

    with pytest.raises(ValueError, match="provider returned") as raised:
        compute_lime_confidence_intervals(
            explainer,
            np.zeros(2),
            failing_predict,
            num_samples=10,
            num_features=1,
            confidence=0.9,
            n_bootstrap=3,
            seed=3,
            label=None,
        )

    assert raised.value is error
