"""Runtime guard for a known goodpoints heap-corruption bug in ctt()/actt().

goodpoints.ctt.ctt()/actt() call gaussianc.sum_gaussian_kernel_by_bin(), which
re-derives num_bins1/num_bins2 from the combined size of the two
(already-compressed) input arrays instead of using the values ctt()/actt()
already computed correctly. This can write out of bounds of the
caller-allocated output matrix whenever the two datasets' KT-Compress
coresets end up with different per-bin sizes -- generic whenever the
reference and current sample sizes differ substantially -- corrupting the
heap and eventually crashing the process.

apply_goodpoints_patches() wraps goodpoints.ctt.ctt/actt so that requests
hitting this precondition raise a normal ValueError instead of corrupting
memory. This is a stopgap: MMD ctt/actt becomes unavailable for the
unbalanced-size case until a goodpoints release containing the real fix
(see SudipSinha/goodpoints, branch fix/unbalanced-sample-size-heap-corruption)
is installed -- at which point this guard detects the fix and disables itself.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_trustyai_patched"


class GoodpointsPatchError(Exception):
    """Raised when the goodpoints patch target is missing or has changed shape."""


def apply_goodpoints_patches() -> None:
    """Idempotently guard goodpoints.ctt.ctt/actt against the known bug.

    Safe to call when goodpoints isn't installed at all (it's an optional
    'mmd' extra) and safe to call multiple times. Raises GoodpointsPatchError
    if goodpoints IS installed but ctt.ctt/ctt.actt don't exist, so a future
    goodpoints release that restructures the module fails loudly instead of
    silently leaving the service unprotected.
    """
    try:
        from goodpoints import ctt as goodpoints_ctt  # noqa: PLC0415
        from goodpoints import gaussianc  # noqa: PLC0415
    except ImportError:
        return  # goodpoints not installed; nothing to protect

    sig = inspect.signature(gaussianc.sum_gaussian_kernel_by_bin)
    if "num_bins1" in sig.parameters:
        logger.info(
            "goodpoints.gaussianc.sum_gaussian_kernel_by_bin already accepts "
            "num_bins1/num_bins2; skipping trustyai-service compatibility guard."
        )
        return

    for name in ("ctt", "actt"):
        if not hasattr(goodpoints_ctt, name):
            msg = (
                f"goodpoints.ctt.{name} not found; cannot apply the "
                "unbalanced-sample-size compatibility guard. MMD ctt/actt "
                "drift detection is UNSAFE (may crash) until this guard is "
                "updated for the installed goodpoints version."
            )
            raise GoodpointsPatchError(msg)

    if getattr(goodpoints_ctt.ctt, _PATCHED_ATTR, False):
        return  # already patched this process

    goodpoints_ctt.ctt = _guard(goodpoints_ctt.ctt)
    goodpoints_ctt.actt = _guard(goodpoints_ctt.actt)
    logger.warning(
        "Applied trustyai-service compatibility guard for goodpoints.ctt.ctt/actt "
        "(unbalanced-sample-size heap-corruption bug). Remove once a goodpoints "
        "release containing the upstream fix is installed."
    )


def _guard(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ctt()/actt() to reject inputs that would trigger the known bug."""

    def guarded(
        x1: Any,  # noqa: ANN401
        x2: Any,  # noqa: ANN401
        g: int,
        *args: Any,  # noqa: ANN401
        s: int = 16,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        _raise_if_unbalanced(x1.shape[0], x2.shape[0], s, g)
        return original(x1, x2, g, *args, s=s, **kwargs)

    setattr(guarded, _PATCHED_ATTR, True)
    return guarded


def _raise_if_unbalanced(n1: int, n2: int, s: int, g: int) -> None:
    """Raise ValueError if (n1, n2, s, g) would trigger the known crash.

    Replicates ctt()/actt()'s own num_bins1/num_bins2 derivation, then
    predicts each dataset's compressed per-bin coreset size using a
    deterministic, side-effect-free replica of compress_kt's size formula
    (no randomness is involved in the size, only in which points are
    selected). The bug is triggered precisely when these predicted per-bin
    sizes differ between the two datasets.
    """
    num_bins_total = min(2 * s, n1 + n2)
    bin_size = (n1 + n2) // num_bins_total
    num_bins1 = n1 // bin_size
    num_bins2 = num_bins_total - num_bins1

    if num_bins1 <= 0 or num_bins2 <= 0:
        msg = (
            f"Unsupported sample-size combination for MMD ctt/actt "
            f"(reference={n1}, current={n2}, s={s}): produces "
            f"num_bins1={num_bins1}, num_bins2={num_bins2}."
        )
        raise ValueError(msg)

    bin_out_size1 = _predict_compress_kt_output_size(n1, num_bins1, g) // num_bins1
    bin_out_size2 = _predict_compress_kt_output_size(n2, num_bins2, g) // num_bins2
    if bin_out_size1 != bin_out_size2:
        msg = (
            f"Unsupported sample-size combination for MMD ctt/actt "
            f"(reference={n1}, current={n2}): triggers a known goodpoints "
            "heap-corruption bug. Adjust batch sizes or wait for an updated "
            "goodpoints release."
        )
        raise ValueError(msg)


def _predict_compress_kt_output_size(n: int, num_bins: int, g: int) -> int:
    """Deterministic size-only replica of compress_kt's output-length calculation."""
    n_per_bin = n / num_bins
    nearest_pow_four = 4 ** ((int(n_per_bin).bit_length() - 1) // 2)
    new_n = nearest_pow_four * num_bins if nearest_pow_four != n_per_bin else n
    return min(new_n, int((new_n * num_bins) ** 0.5 * (2**g)))
