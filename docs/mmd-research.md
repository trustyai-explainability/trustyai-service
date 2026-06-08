# MMD Drift Metric: Research & Design

## 1. Background

Maximum Mean Discrepancy (MMD) is a kernel-based two-sample test that detects distributional differences between a reference dataset and current data. Unlike per-feature tests (KS, CompareMeans), MMD is inherently **multivariate** — it computes a single statistic across all features simultaneously.

The Java upstream (`trustyai-explainability`) uses the name "FourierMMD," but this term is not found in the literature — the papers call it "FastMMD" (Zhao & Meng 2014) or "RFF-MMD." We use **MMD** as the metric name, with deprecated `FourierMMD` aliases for backward compatibility.

---

## 2. Algorithm Landscape

### Batch Two-Sample Tests

| Method | Time | Power | Bandwidth-free? | Code |
|--------|------|-------|-----------------|------|
| Exact MMD + permutation (Gretton 2012) | O(B·n²) | Optimal for chosen kernel | No | goodpoints `ctt.ctt(g=0)` / standalone |
| RFF-MMD (Rahimi & Recht 2007) | O(r·n·B) | Moderate (finite r) | No | goodpoints `ctt.rff()` |
| CTT (Domingo-Enrich 2023) | Near-linear | **Optimal** (no approx error) | No | goodpoints `ctt.ctt()` |
| ACTT (CTT + MMDAgg) | O(K·B·n²) | Optimal (adaptive) | **Yes** | goodpoints `ctt.actt()` |
| MMDAgg (Schrab 2023) | O(K·B·n²) | Optimal (adaptive) | **Yes** | goodpoints `ctt.actt()` / standalone |
| MMDAggInc (Schrab 2022) | O(r·n·K·B) | High (linear-time) | **Yes** | — |
| Cross-MMD (Shekhar 2022) | O(n²) | Optimal | No | — |
| Block MMD (Zaremba 2013) | O(d·n^1.5) | High | No | — (superseded by CTT) |

### Online / Streaming

| Method | Time per obs | Code |
|--------|-------------|------|
| Online RFF-MMD (Kalinke 2025) | O(r log n) | [FlopsKa/rff-change-detection](https://github.com/FlopsKa/rff-change-detection) |

### Key Weaknesses of RFF-MMD (FastMMD)

1. Pointwise inconsistent with finite features — can miss drift
2. Matching exact MMD power requires O(n³) features — worse than exact (Domingo-Enrich 2023)
3. Bandwidth-sensitive — no adaptive selection

### MMDAgg Algorithm Summary

Solves bandwidth selection by aggregating tests over K kernels/bandwidths with bisection-based level correction:

1. Compute K kernel matrices from an exponential bandwidth grid
2. Run B1+B2 wild bootstrap (m=n) or permutation (m≠n) samples
3. Bisection over B3 iterations to find correction parameter u controlling Type I error at level α
4. Reject H₀ if any kernel's observed statistic exceeds its corrected threshold

---

## 3. Library Evaluation

### goodpoints (Microsoft) — **Selected as backend**

| Aspect | Assessment |
|--------|-----------|
| License | MIT |
| Maintainer | Lester Mackey (Microsoft Research), sole active contributor |
| Tests / CI | None |
| Type hints | None |
| Input validation | Kernel name only — wrong shapes cause silent Cython corruption |
| NumPy 2.x | Fixed via [PR #10](https://github.com/microsoft/goodpoints/pull/10) (SudipSinha), not yet on PyPI |
| Wheels | macOS only on PyPI; Linux must build from source |
| Debug prints | `print()`/`printf()` in production code (especially `actt()` bisection) |
| Kernel support | Gaussian only |

**Available functions:** `ctt.ctt()` (CTT), `ctt.rff()` (RFF), `ctt.actt()` (aggregated CTT), `ctt.lrctt()` (low-rank CTT).

**Known bug:** CTT segfaults on edge-case sample sizes where `bin_size = max(1, (n1+n2)//(2*s))` doesn't evenly divide n1 and n2. Workaround: use sample sizes that are multiples of 2·s (default s=16).

**Kernel extensibility:** The compression pipeline (`compressc.pyx`) uses function pointer typedefs — adding kernels follows the existing Sobolev pattern. The CTT pipeline (`gaussianc.pyx`) has no abstraction layer and would need parallel Cython functions or a refactor to function pointers. RFF frequency sampling is pure Python — trivial to branch. A medium-sized upstream PR, not an architectural change.

**Why selected:** CTT's kernel thinning (29KB optimized Cython) cannot be trivially reimplemented. Near-linear time with no approximation error is a genuine advantage.

### antoninschrab/mmdagg & agginc — **Not viable**

| Issue | mmdagg | agginc |
|-------|--------|--------|
| PyPI | Not published | Not published |
| Tests / CI | None | None |
| `__init__.py` | Crashes without JAX | Crashes without JAX |
| Dependencies | numpy, scipy | numpy, scipy, psutil, gputil (abandoned 2019) |
| API return type | `int` (0/1) | `int` (0/1) |
| Random API | Deprecated `RandomState` | Deprecated `RandomState` |
| Bugs | `np.testing.assert_array_equal` in prod code | Missing `import warnings` → runtime crash |

Frozen research artifacts (last commits: 2024-04, 2023-04). Not production-viable.

### Other libraries

| Library | Verdict |
|---------|---------|
| alibi-detect | Requires TensorFlow or PyTorch — too heavy |
| TorchDrift | Requires PyTorch — too heavy |
| kernel_two_sample_test | Unmaintained, reference only |

---

## 4. Design Decisions

### Backend: goodpoints

We wrap `ctt.ctt()`, `ctt.rff()`, and `ctt.actt()` behind our `MMD.compute(method=...)` dispatcher, adding input validation, type hints (`MMDResult` TypedDict), and error handling. goodpoints is an optional dependency (`trustyai-service[mmd]`), pinned to GitHub `main` until the NumPy fix reaches PyPI.

### Method dispatch

`MMD.compute()` uses a `method` parameter (like `scipy.optimize.minimize`). Common parameters (`alpha`, `seed`) are explicit; method-specific ones pass through `**kwargs`.

| Method | Algorithm | Key kwargs |
|--------|-----------|-----------|
| `"ctt"` (default) | Compress Then Test | `bandwidth`, `kernel`, `num_permutations`, `compression_level`, `num_bins` |
| `"rff"` | Random Fourier Features | `bandwidth`, `kernel`, `num_permutations`, `num_features` |
| `"actt"` | Aggregated CTT | `bandwidth`, `kernel`, `num_permutations`, `num_bandwidths`, `b2`, `b3` |

### Gaussian-only limitation

goodpoints only supports the Gaussian kernel. Multi-kernel aggregation (Laplace, IMQ, Matérn) requires an upstream PR to goodpoints. The function pointer pattern already exists in `compressc.pyx`, so this is feasible.

---

## 5. Known Caveats

- **Gaussian kernel only** — goodpoints does not support Laplace, Matérn, or other kernels
- **CTT segfault** — sample sizes must satisfy `bin_size | n1` and `bin_size | n2` where `bin_size = max(1, (n1+n2)//(2*s))`
- **Debug prints** — `actt()` emits verbose bisection output to stdout
- **No type stubs** — basedpyright reports `Unknown` types for all goodpoints returns
- **PyPI lag** — NumPy 2.x fix not yet released; must install from Git

---

## 6. References

1. Gretton, Borgwardt, Rasch, Schölkopf & Smola. "A Kernel Two-Sample Test." JMLR 13, 2012.
2. Zhao & Meng. "FastMMD: Ensemble of Circular Discrepancy for Efficient Two-Sample Test." arXiv:1405.2664, 2014.
3. Rahimi & Recht. "Random Features for Large-Scale Kernel Machines." NeurIPS, 2007.
4. Zaremba, Gretton & Blaschko. "B-test: A Non-parametric, Low Variance Kernel Two-Sample Test." NeurIPS, 2013.
5. Liu et al. "Learning Deep Kernels for Non-Parametric Two-Sample Tests." ICML, 2020.
6. Shekhar, Kim & Ramdas. "Permutation-Free Kernel Two-Sample Testing." arXiv:2211.14908, NeurIPS, 2022.
7. Schrab, Kim, Albert, Laurent, Guedj & Gretton. "Efficient Aggregated Kernel Tests using Incomplete U-statistics." arXiv:2206.09194, NeurIPS, 2022.
8. Schrab, Kim, Albert, Laurent, Guedj & Gretton. "MMD Aggregated Two-Sample Test." arXiv:2110.15073, JMLR 24(194), 2023.
9. Domingo-Enrich, Dwivedi & Mackey. "Compress Then Test: Powerful Kernel Testing in Near-Linear Time." arXiv:2301.05974, AISTATS, 2023.
10. Bodenham & Kawahara. "euMMD: Efficiently Computing the MMD in O(N log N) Time." Statistics and Computing, 2023.
11. Choi & Kim. "Computational-Statistical Trade-off in Kernel Two-Sample Testing with RFF." arXiv:2407.08976, 2024.
12. Mukherjee et al. "Minimax Optimal Kernel Two-Sample Tests with Random Features." arXiv:2502.20755, 2025.
13. Kalinke & Gavioli-Akilagun. "Optimal Online Change Detection via Random Fourier Features." arXiv:2505.17789, 2025.
