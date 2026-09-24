"""Tests for the pure callable-based KernelSHAP core."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import trustyai_service.core.explainers.local.shap as shap_core
from trustyai_service.core.explainers.local.shap import (
    ShapExplanationResult,
    compute_confidence_intervals,
    compute_shap_result,
)


def _background_and_instance() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic background and one explained instance."""
    background = np.asarray(
        [
            [-2.0, 0.5, 1.0],
            [-1.0, 1.0, 0.0],
            [0.0, -0.5, 2.0],
            [1.0, 1.5, -1.0],
            [2.0, -1.0, 0.5],
            [3.0, 0.0, -2.0],
            [4.0, 2.0, 1.5],
            [5.0, -1.5, -0.5],
        ],
        dtype=float,
    )
    instance = np.asarray([1.25, 0.25, 0.75], dtype=float)
    return background, instance


def _regression_predict(values: np.ndarray) -> np.ndarray:
    """Return one deterministic regression value per row."""
    return 2.0 * values[:, 0] - values[:, 1] + 0.5 * values[:, 2]


def _positive_probability(values: np.ndarray) -> np.ndarray:
    """Return a selected positive-class probability per row."""
    score = 0.25 * values[:, 0] - 0.5 * values[:, 1] + values[:, 2]
    return 1.0 / (1.0 + np.exp(-score))


def _binary_probabilities(values: np.ndarray) -> np.ndarray:
    """Return a two-column classification probability matrix."""
    positive = _positive_probability(values)
    return np.column_stack((1.0 - positive, positive))


def _selected_positive_class(values: np.ndarray) -> np.ndarray:
    """Select class 1 from the raw binary probability matrix."""
    return _binary_probabilities(values)[:, 1]


def test_core_signatures_use_only_the_shared_callable_and_keyword_options() -> None:
    """Expose the exact planned positional and keyword-only arguments."""
    result_signature = inspect.signature(compute_shap_result)
    confidence_signature = inspect.signature(compute_confidence_intervals)

    assert [
        parameter.name
        for parameter in result_signature.parameters.values()
        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD
    ] == ["background", "instance", "predict_fn"]
    assert [
        parameter.name
        for parameter in confidence_signature.parameters.values()
        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD
    ] == ["background", "instance", "predict_fn"]

    for name in (
        "instance_prediction",
        "n_samples",
        "link",
        "l1_reg",
    ):
        assert result_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    for name in (
        "instance_prediction",
        "n_samples",
        "link",
        "l1_reg",
        "confidence",
        "n_bootstrap",
        "seed",
    ):
        assert (
            confidence_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        )
    assert confidence_signature.parameters["n_bootstrap"].default == 50
    assert confidence_signature.parameters["seed"].default is inspect.Parameter.empty


def test_importing_core_does_not_load_optional_or_service_integrations() -> None:
    """Importing the core must not import SHAP or service-layer integrations."""
    repository_root = Path(__file__).parents[4]
    script = """
import builtins
import sys

sys.path.insert(0, "src")
forbidden = (
    "shap",
    "sklearn",
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
import trustyai_service.core.explainers.local.shap
"""

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not shap_core._SHAP_AVAILABLE, reason="SHAP extra is unavailable")
def test_regression_identity_result_is_additive_and_has_one_value_per_feature() -> None:
    """Identity-link regression values explain the supplied instance prediction."""
    background, instance = _background_and_instance()
    prediction = float(_regression_predict(instance.reshape(1, -1))[0])

    result = compute_shap_result(
        background,
        instance,
        _regression_predict,
        instance_prediction=prediction,
        n_samples=40,
        link="identity",
        l1_reg="bic",
    )

    assert isinstance(result, ShapExplanationResult)
    assert result.values.shape == (background.shape[1],)
    assert np.isfinite(result.values).all()
    assert result.linked_prediction == pytest.approx(prediction)
    assert result.base_value + float(result.values.sum()) == pytest.approx(
        result.linked_prediction,
        abs=1e-5,
    )


@pytest.mark.skipif(not shap_core._SHAP_AVAILABLE, reason="SHAP extra is unavailable")
@pytest.mark.parametrize("link", ["identity", "logit"])
def test_selected_class_probability_supports_both_shap_links(link: str) -> None:
    """A selected class callable is explained in raw or logit probability units."""
    background, instance = _background_and_instance()
    selected_prediction = float(_selected_positive_class(instance.reshape(1, -1))[0])

    result = compute_shap_result(
        background,
        instance,
        _selected_positive_class,
        instance_prediction=selected_prediction,
        n_samples=40,
        link=link,
        l1_reg="num_features(2)",
    )

    expected_prediction = (
        selected_prediction
        if link == "identity"
        else float(np.log(selected_prediction / (1.0 - selected_prediction)))
    )
    assert result.values.shape == (background.shape[1],)
    assert result.linked_prediction == pytest.approx(expected_prediction)
    assert result.base_value + float(result.values.sum()) == pytest.approx(
        result.linked_prediction,
        abs=1e-5,
    )


@pytest.mark.skipif(not shap_core._SHAP_AVAILABLE, reason="SHAP extra is unavailable")
def test_one_column_positive_probability_callable_is_accepted_when_already_selected() -> (
    None
):
    """The core accepts an explicitly selected one-column positive probability."""
    background, instance = _background_and_instance()
    selected_prediction = float(_positive_probability(instance.reshape(1, -1))[0])

    result = compute_shap_result(
        background,
        instance,
        lambda values: _binary_probabilities(values)[:, 1],
        instance_prediction=selected_prediction,
        n_samples=30,
        link="identity",
        l1_reg=0.0,
    )

    assert result.values.shape == (background.shape[1],)
    assert np.isfinite(result.values).all()


@pytest.mark.skipif(not shap_core._SHAP_AVAILABLE, reason="SHAP extra is unavailable")
@pytest.mark.parametrize("l1_reg", ["aic", "bic", "num_features(2)", 0.0])
def test_supported_regularizers_return_finite_attributions(l1_reg: str | float) -> None:
    """String and numeric SHAP regularizers are passed through to KernelExplainer."""
    background, instance = _background_and_instance()
    prediction = float(_regression_predict(instance.reshape(1, -1))[0])

    result = compute_shap_result(
        background,
        instance,
        _regression_predict,
        instance_prediction=prediction,
        n_samples=40,
        link="identity",
        l1_reg=l1_reg,
    )

    assert result.values.shape == (background.shape[1],)
    assert np.isfinite(result.values).all()


@pytest.mark.skipif(not shap_core._SHAP_AVAILABLE, reason="SHAP extra is unavailable")
def test_confidence_intervals_are_seeded_and_bracket_the_point_estimate() -> None:
    """Bootstrap bounds are reproducible and have the expected feature shape."""
    background, instance = _background_and_instance()
    prediction = float(_regression_predict(instance.reshape(1, -1))[0])
    kwargs = {
        "instance_prediction": prediction,
        "n_samples": 30,
        "link": "identity",
        "l1_reg": "bic",
        "confidence": 0.9,
        "n_bootstrap": 6,
        "seed": 11,
    }

    first = compute_confidence_intervals(
        background,
        instance,
        _regression_predict,
        **kwargs,
    )
    second = compute_confidence_intervals(
        background,
        instance,
        _regression_predict,
        **kwargs,
    )

    lower, upper = first
    assert lower is not None
    assert upper is not None
    np.testing.assert_array_equal(lower, second[0])
    np.testing.assert_array_equal(upper, second[1])
    assert lower.shape == (background.shape[1],)
    assert upper.shape == (background.shape[1],)
    assert np.all(lower <= upper)


def test_confidence_one_disables_bootstrap() -> None:
    """The shared confidence sentinel avoids optional SHAP work."""
    background, instance = _background_and_instance()

    lower, upper = compute_confidence_intervals(
        background,
        instance,
        _regression_predict,
        instance_prediction=0.0,
        n_samples=10,
        link="identity",
        l1_reg="bic",
        confidence=1.0,
        seed=None,
    )

    assert lower is None
    assert upper is None


def test_provider_error_identity_is_preserved() -> None:
    """A primary callable failure is not replaced by a core-specific error."""
    background, instance = _background_and_instance()
    error = RuntimeError("provider failed")

    def failing_predict(_: np.ndarray) -> np.ndarray:
        raise error

    with pytest.raises(RuntimeError) as raised:
        compute_shap_result(
            background,
            instance,
            failing_predict,
            instance_prediction=0.5,
            n_samples=20,
            link="identity",
            l1_reg="bic",
        )

    assert raised.value is error


@pytest.mark.skipif(not shap_core._SHAP_AVAILABLE, reason="SHAP extra is unavailable")
def test_bootstrap_provider_error_identity_is_preserved() -> None:
    """Bootstrap skipping never swallows a provider callable failure."""
    background, instance = _background_and_instance()
    error = ValueError("provider returned an invalid response")

    def failing_predict(_: np.ndarray) -> np.ndarray:
        raise error

    with pytest.raises(ValueError, match="provider returned") as raised:
        compute_confidence_intervals(
            background,
            instance,
            failing_predict,
            instance_prediction=0.5,
            n_samples=20,
            link="identity",
            l1_reg="bic",
            confidence=0.9,
            n_bootstrap=3,
            seed=4,
        )

    assert raised.value is error


def test_supplied_instance_prediction_is_used_without_a_singleton_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The result uses the supplied prediction and does not call the callable itself."""
    background, instance = _background_and_instance()
    calls: list[np.ndarray] = []

    class FakeLink:
        def f(self, values: object) -> np.ndarray:
            return np.asarray(values, dtype=float)

    class FakeKernelExplainer:
        def __init__(self, model: object, data: object, *, link: str) -> None:
            del model, data
            assert link == "identity"
            self.expected_value = 0.25
            self.link = FakeLink()

        def shap_values(self, values: np.ndarray, **kwargs: object) -> np.ndarray:
            assert values.shape == (1, 3)
            assert kwargs == {
                "nsamples": 10,
                "l1_reg": "bic",
                "silent": True,
            }
            return np.zeros((1, 3))

    class FakeShap:
        KernelExplainer = FakeKernelExplainer

        @staticmethod
        def kmeans(values: np.ndarray, count: int) -> np.ndarray:
            assert values is background
            assert count == 4
            return values[:count]

    def predict(values: np.ndarray) -> np.ndarray:
        calls.append(values.copy())
        return _regression_predict(values)

    fake_shap = FakeShap()

    def load_fake_shap() -> FakeShap:
        return fake_shap

    monkeypatch.setattr(shap_core, "_load_shap_module", load_fake_shap)
    result = compute_shap_result(
        background,
        instance,
        predict,
        instance_prediction=9.0,
        n_samples=10,
        link="identity",
        l1_reg="bic",
    )

    assert calls == []
    assert result.linked_prediction == 9.0


def test_optional_dependency_failure_is_explicit_and_endpoint_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing SHAP dependency raises a dedicated core error."""
    background, instance = _background_and_instance()
    monkeypatch.setattr(shap_core, "_SHAP_AVAILABLE", False)

    with pytest.raises(shap_core.SHAPUnavailableError, match="SHAP dependency"):
        compute_shap_result(
            background,
            instance,
            _regression_predict,
            instance_prediction=0.0,
            n_samples=10,
            link="identity",
            l1_reg="bic",
        )


def test_zero_feature_inputs_are_rejected_before_shap_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a tabular contract with no features before optional work starts."""
    background = np.empty((3, 0))
    instance = np.empty(0)

    def unexpected_loader() -> object:
        pytest.fail("SHAP must not load for zero-feature input")

    monkeypatch.setattr(shap_core, "_load_shap_module", unexpected_loader)

    with pytest.raises(ValueError, match="at least one feature"):
        compute_shap_result(
            background,
            instance,
            _regression_predict,
            instance_prediction=0.0,
            n_samples=10,
            link="identity",
            l1_reg="bic",
        )

    with pytest.raises(ValueError, match="at least one feature"):
        compute_confidence_intervals(
            background,
            instance,
            _regression_predict,
            instance_prediction=0.0,
            n_samples=10,
            link="identity",
            l1_reg="bic",
            confidence=0.9,
            n_bootstrap=1,
            seed=1,
        )


def test_finiteness_is_checked_after_float_conversion_for_object_overflow() -> None:
    """Reject object values that coerce to infinity as float64."""
    background = np.asarray([["1e400"]], dtype=object)
    instance = np.asarray(["1"], dtype=object)

    with pytest.raises(ValueError, match="finite"):
        shap_core._validate_inputs(background, instance)


def test_finiteness_is_checked_after_float_conversion_for_longdouble_overflow() -> None:
    """Reject finite extended-precision values that overflow float64 conversion."""
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("longdouble has no wider range than float64 on this platform")

    value = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    assert np.isfinite(value)

    with pytest.raises(ValueError, match="finite"):
        shap_core._validate_inputs(
            np.asarray([[value]], dtype=np.longdouble),
            np.asarray([value], dtype=np.longdouble),
        )


@pytest.mark.parametrize(
    "l1_reg",
    [
        "bogus",
        "num_features(x)",
        "num_features(0)",
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
    ],
)
def test_invalid_l1_regularizers_are_rejected_before_shap_is_loaded(
    l1_reg: str | float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid SHAP regularizers are configuration errors, not bootstrap misses."""
    background, instance = _background_and_instance()

    def unexpected_loader() -> object:
        pytest.fail("invalid l1_reg must be rejected before SHAP loads")

    monkeypatch.setattr(shap_core, "_load_shap_module", unexpected_loader)

    with pytest.raises((TypeError, ValueError)):
        compute_shap_result(
            background,
            instance,
            _regression_predict,
            instance_prediction=1.0,
            n_samples=10,
            link="identity",
            l1_reg=l1_reg,
        )

    with pytest.raises((TypeError, ValueError)):
        compute_confidence_intervals(
            background,
            instance,
            _regression_predict,
            instance_prediction=1.0,
            n_samples=10,
            link="identity",
            l1_reg=l1_reg,
            confidence=0.9,
            n_bootstrap=1,
            seed=1,
        )


def test_bootstrap_skips_only_numpy_linear_algebra_degeneracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip a singular bootstrap solve while retaining successful samples."""
    background, instance = _background_and_instance()
    calls = 0

    def fake_compute(
        *_args: object,
        **_kwargs: object,
    ) -> ShapExplanationResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            error = "singular solve"
            raise np.linalg.LinAlgError(error)
        return ShapExplanationResult(
            values=np.asarray([1.0, 2.0, 3.0]),
            base_value=0.0,
            linked_prediction=0.0,
        )

    monkeypatch.setattr(shap_core, "_compute_shap_result", fake_compute)
    lower, upper = compute_confidence_intervals(
        background,
        instance,
        _regression_predict,
        instance_prediction=1.0,
        n_samples=10,
        link="identity",
        l1_reg="bic",
        confidence=0.9,
        n_bootstrap=2,
        seed=1,
    )

    assert lower is not None
    assert upper is not None
    np.testing.assert_array_equal(lower, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(upper, [1.0, 2.0, 3.0])


def test_all_numpy_linear_algebra_bootstrap_failures_return_empty_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only all-numerical bootstrap degeneracy may produce empty intervals."""
    error = "singular solve"

    def degenerate_compute(
        *_args: object,
        **_kwargs: object,
    ) -> ShapExplanationResult:
        raise np.linalg.LinAlgError(error)

    monkeypatch.setattr(shap_core, "_compute_shap_result", degenerate_compute)
    background, instance = _background_and_instance()

    lower, upper = compute_confidence_intervals(
        background,
        instance,
        _regression_predict,
        instance_prediction=1.0,
        n_samples=10,
        link="identity",
        l1_reg="bic",
        confidence=0.9,
        n_bootstrap=2,
        seed=1,
    )

    assert lower is None
    assert upper is None


def test_malformed_shap_result_is_not_treated_as_bootstrap_degeneracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed SHAP output must propagate instead of yielding empty intervals."""
    background, instance = _background_and_instance()

    class FakeLink:
        def f(self, values: object) -> object:
            return values

    class FakeKernelExplainer:
        expected_value = 0.0
        link = FakeLink()

        def __init__(self, model: object, data: object, *, link: str) -> None:
            del model, data, link

        def shap_values(self, values: np.ndarray, **kwargs: object) -> np.ndarray:
            del values, kwargs
            return np.zeros((1, 2))

    class FakeShap:
        KernelExplainer = FakeKernelExplainer

        @staticmethod
        def kmeans(values: np.ndarray, count: int) -> np.ndarray:
            return values[:count]

    def load_fake_shap() -> FakeShap:
        return FakeShap()

    monkeypatch.setattr(shap_core, "_load_shap_module", load_fake_shap)

    with pytest.raises(RuntimeError, match="malformed"):
        compute_confidence_intervals(
            background,
            instance,
            _regression_predict,
            instance_prediction=1.0,
            n_samples=10,
            link="identity",
            l1_reg="bic",
            confidence=0.9,
            n_bootstrap=1,
            seed=1,
        )


def test_bootstrap_numpy_linear_algebra_provider_error_identity_is_preserved() -> None:
    """A provider LinAlgError is not confused with SHAP numerical degeneracy."""
    background, instance = _background_and_instance()
    error = np.linalg.LinAlgError("provider failed")

    def failing_predict(_: np.ndarray) -> np.ndarray:
        raise error

    with pytest.raises(np.linalg.LinAlgError) as raised:
        compute_confidence_intervals(
            background,
            instance,
            failing_predict,
            instance_prediction=0.5,
            n_samples=20,
            link="identity",
            l1_reg="bic",
            confidence=0.9,
            n_bootstrap=1,
            seed=4,
        )

    assert raised.value is error


@pytest.mark.parametrize(
    "error",
    [
        ImportError("missing dependency"),
        AttributeError("incompatible SHAP API"),
        RuntimeError("incompatible SHAP runtime"),
        OSError("incompatible SHAP binary"),
    ],
)
def test_incompatible_shap_import_errors_are_normalized(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional SHAP initialization failures use one endpoint-neutral error."""
    background, instance = _background_and_instance()
    monkeypatch.setattr(shap_core, "_SHAP_AVAILABLE", True)

    def failing_import(_: str) -> object:
        raise error

    monkeypatch.setattr(shap_core, "import_module", failing_import)

    with pytest.raises(shap_core.SHAPUnavailableError) as raised:
        compute_shap_result(
            background,
            instance,
            _regression_predict,
            instance_prediction=1.0,
            n_samples=10,
            link="identity",
            l1_reg="bic",
        )

    assert str(raised.value) == "SHAP dependency is unavailable"
    assert raised.value.__cause__ is error


def test_incompatible_loaded_shap_api_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loaded module without the required KernelSHAP API is unavailable."""
    background, instance = _background_and_instance()

    def load_empty_shap() -> object:
        return object()

    monkeypatch.setattr(shap_core, "_load_shap_module", load_empty_shap)

    with pytest.raises(shap_core.SHAPUnavailableError):
        compute_shap_result(
            background,
            instance,
            _regression_predict,
            instance_prediction=1.0,
            n_samples=10,
            link="identity",
            l1_reg="bic",
        )


def test_incompatible_shap_constructor_api_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incompatible KernelExplainer constructor uses the stable unavailable error."""
    background, instance = _background_and_instance()

    class FakeShap:
        @staticmethod
        def kmeans(values: np.ndarray, count: int) -> np.ndarray:
            return values[:count]

        @staticmethod
        def constructor(
            model: object,
            data: np.ndarray,
            *,
            link: str,
        ) -> object:
            del model, data, link
            error = "incompatible constructor"
            raise TypeError(error)

        KernelExplainer = staticmethod(constructor)

    def load_fake_shap() -> FakeShap:
        return FakeShap()

    monkeypatch.setattr(shap_core, "_load_shap_module", load_fake_shap)

    with pytest.raises(shap_core.SHAPUnavailableError):
        compute_shap_result(
            background,
            instance,
            _regression_predict,
            instance_prediction=1.0,
            n_samples=10,
            link="identity",
            l1_reg="bic",
        )
