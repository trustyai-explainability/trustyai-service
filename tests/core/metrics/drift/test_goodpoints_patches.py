"""Tests for the goodpoints unbalanced-sample-size compatibility guard."""

import builtins
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
from goodpoints import compress, gaussianc
from goodpoints import ctt as goodpoints_ctt

from trustyai_service.core.metrics.drift._goodpoints_patches import (
    GoodpointsPatchError,
    _predict_compress_kt_output_size,
    apply_goodpoints_patches,
)


@pytest.fixture(autouse=True)
def _reset_goodpoints_ctt_patch() -> Generator[None]:
    """Ensure each test starts from goodpoints' real, unpatched ctt/actt."""
    original_ctt = goodpoints_ctt.ctt
    original_actt = goodpoints_ctt.actt
    yield
    goodpoints_ctt.ctt = original_ctt
    goodpoints_ctt.actt = original_actt


class TestApplyGoodpointsPatches:
    """Tests for apply_goodpoints_patches()."""

    def test_guard_raises_for_known_crash_shape(self) -> None:
        """The guard rejects the exact TrustyAI drift-detection crash shape."""
        apply_goodpoints_patches()
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal((2000, 4))
        x2 = rng.standard_normal((100, 4))
        with pytest.raises(ValueError, match=r"triggers a known goodpoints"):
            goodpoints_ctt.ctt(
                x1, x2, 0, B=39, s=16, lam=1.0, kernel="gauss", alpha=0.05
            )

    def test_guard_allows_balanced_shapes(self) -> None:
        """The guard lets balanced (non-crashing) ctt() calls through."""
        apply_goodpoints_patches()
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal((64, 2))
        x2 = rng.standard_normal((64, 2))
        result = goodpoints_ctt.ctt(x1, x2, 0, B=9, s=4, null_seed=0, statistic_seed=1)
        assert np.isfinite(result.statistic_values)

    def test_guard_allows_balanced_actt_shapes(self) -> None:
        """The guard lets balanced (non-crashing) actt() calls through."""
        apply_goodpoints_patches()
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal((256, 2))
        x2 = rng.standard_normal((256, 2))
        result = goodpoints_ctt.actt(
            x1,
            x2,
            0,
            B=9,
            B_2=5,
            B_3=3,
            s=8,
            lam=np.array([1.0]),
            null_seed=0,
            statistic_seed=1,
        )
        assert np.isfinite(next(iter(result.statistic_values.values())))

    def test_guard_raises_for_known_crash_shape_actt(self) -> None:
        """The guard rejects the crash shape for actt() as well as ctt()."""
        apply_goodpoints_patches()
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal((2000, 2))
        x2 = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match=r"triggers a known goodpoints"):
            goodpoints_ctt.actt(
                x1, x2, 0, B=49, B_2=30, B_3=5, s=16, lam=np.array([1.0])
            )

    def test_guard_raises_value_error_for_zero_bin_count(self) -> None:
        """A tiny reference vs. large current sample doesn't crash with ZeroDivisionError.

        n1=1 with the default s=16 makes num_bins1 derive to 0, which would
        otherwise raise ZeroDivisionError inside the guard itself instead of
        the intended clean ValueError.
        """
        apply_goodpoints_patches()
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal((1, 2))
        x2 = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match=r"num_bins1=0"):
            goodpoints_ctt.ctt(x1, x2, 0, B=9, s=16, null_seed=0, statistic_seed=0)

    def test_patch_is_idempotent(self) -> None:
        """Applying the patch twice does not re-wrap an already-guarded function."""
        apply_goodpoints_patches()
        patched_ctt = goodpoints_ctt.ctt
        patched_actt = goodpoints_ctt.actt
        apply_goodpoints_patches()
        assert goodpoints_ctt.ctt is patched_ctt
        assert goodpoints_ctt.actt is patched_actt

    def test_patch_skips_when_already_fixed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard is a no-op once the installed goodpoints has the real fix."""

        def fake_sum_gaussian_kernel_by_bin(  # noqa: PLR0913
            x1: Any,  # noqa: ANN401
            x2: Any,  # noqa: ANN401
            lam_sqd: Any,  # noqa: ANN401
            num_bins1: Any,  # noqa: ANN401
            num_bins2: Any,  # noqa: ANN401
            k_sum: Any,  # noqa: ANN401
        ) -> None:
            pass

        monkeypatch.setattr(
            gaussianc, "sum_gaussian_kernel_by_bin", fake_sum_gaussian_kernel_by_bin
        )
        original_ctt = goodpoints_ctt.ctt
        apply_goodpoints_patches()
        assert goodpoints_ctt.ctt is original_ctt

    def test_patch_raises_when_ctt_attr_missing(self) -> None:
        """A restructured goodpoints (missing ctt.ctt) fails loudly, not silently."""
        original_ctt = goodpoints_ctt.ctt
        del goodpoints_ctt.ctt
        try:
            with pytest.raises(GoodpointsPatchError, match=r"goodpoints\.ctt\.ctt"):
                apply_goodpoints_patches()
        finally:
            goodpoints_ctt.ctt = original_ctt

    def test_patch_noop_when_goodpoints_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environments without goodpoints installed are unaffected."""
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if name == "goodpoints" or name.startswith("goodpoints."):
                msg = f"No module named {name!r}"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        apply_goodpoints_patches()  # should not raise


class TestPredictCompressKtOutputSize:
    """Tests for _predict_compress_kt_output_size()."""

    @pytest.mark.parametrize(
        ("n", "num_bins", "g"),
        [
            (2000, 30, 0),
            (100, 2, 0),
            (1000, 29, 0),
            (37, 3, 0),
            (256, 8, 0),
            (64, 4, 0),
            (1920, 30, 1),
        ],
    )
    def test_matches_real_compress_kt(self, n: int, num_bins: int, g: int) -> None:
        """The size prediction matches compress_kt's real output length."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((n, 2))
        actual = len(
            compress.compress_kt(x, b"gaussian", num_bins=num_bins, g=g, seed=0)
        )
        predicted = _predict_compress_kt_output_size(n, num_bins, g)
        assert actual == predicted
