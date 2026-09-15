"""Unified test factories for core explainer computation functions.

Covers the confidence-interval (CI) contract shared by all local explainers:

- ``confidence=1.0`` is a sentinel meaning "skip CI" → returns ``(None, None)``
- ``confidence < 1.0`` returns ``(lower, upper)`` where ``lower[i] <= upper[i]``

The CI return type varies by explainer:
- LIME: ``(dict[str, float] | None, dict[str, float] | None)``
- SHAP: ``(np.ndarray | None, np.ndarray | None)``

Both are handled by the same factories via duck-typing.
"""

from collections.abc import Callable
from typing import Any

import numpy as np


def make_ci_disabled_at_one_test(
    ci_fn: Callable,
    build_args: Callable[[], tuple[tuple, dict[str, Any]]],
) -> Callable[[object], None]:
    """``confidence=1.0`` sentinel returns ``(None, None)`` without computing.

    Args:
        ci_fn: The CI function under test, e.g.
            ``compute_lime_confidence_intervals`` or
            ``compute_confidence_intervals``.
        build_args: Zero-argument callable that returns ``(args, kwargs)``
            to pass to ``ci_fn``.  The factory overrides ``kwargs["confidence"]``
            with ``1.0`` before calling.

    """

    def test_impl(_: object) -> None:
        args, kwargs = build_args()
        kwargs["confidence"] = 1.0
        lower, upper = ci_fn(*args, **kwargs)
        assert lower is None, f"Expected lower=None for confidence=1.0, got {lower!r}"
        assert upper is None, f"Expected upper=None for confidence=1.0, got {upper!r}"

    return test_impl


def make_ci_bounds_valid_test(
    ci_fn: Callable,
    build_args: Callable[[], tuple[tuple, dict[str, Any]]],
    confidence: float = 0.90,
) -> Callable[[object], None]:
    """``confidence < 1.0`` returns non-None bounds with ``lower[i] <= upper[i]``.

    Works for both ``dict`` (LIME) and ``np.ndarray`` (SHAP) return types.

    Args:
        ci_fn: The CI function under test.
        build_args: Zero-argument callable that returns ``(args, kwargs)``.
            The factory overrides ``kwargs["confidence"]`` with ``confidence``.
        confidence: Coverage value to use (must be < 1.0).  Defaults to 0.90.

    """
    assert 0.0 < confidence < 1.0, "confidence must be in (0, 1) for this test"

    def test_impl(_: object) -> None:
        args, kwargs = build_args()
        kwargs["confidence"] = confidence
        lower, upper = ci_fn(*args, **kwargs)

        assert lower is not None, f"Expected non-None lower for confidence={confidence}"
        assert upper is not None, f"Expected non-None upper for confidence={confidence}"

        # Handle dict (LIME) or ndarray (SHAP)
        if isinstance(lower, dict):
            assert lower.keys() == upper.keys(), (
                f"lower keys={list(lower)} != upper keys={list(upper)}"
            )
            for key in lower:
                assert lower[key] <= upper[key], (
                    f"lower[{key!r}]={lower[key]} > upper[{key!r}]={upper[key]}"
                )
        else:
            arr_lower = np.asarray(lower)
            arr_upper = np.asarray(upper)
            assert arr_lower.shape == arr_upper.shape, (
                f"lower.shape={arr_lower.shape} != upper.shape={arr_upper.shape}"
            )
            assert np.all(arr_lower <= arr_upper), (
                f"lower not <= upper elementwise: {arr_lower} vs {arr_upper}"
            )

    return test_impl
