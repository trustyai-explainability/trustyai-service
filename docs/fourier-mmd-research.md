# Fourier MMD Drift Metric: Research Findings

## 1. Naming Analysis

### Problem

Searching for "Fourier MMD" on the web returns results primarily about **Graph Fourier MMD** (Leone et al., 2023), a completely different method based on graph spectral theory. The papers cited for this metric never use the term "Fourier MMD."

### What the Papers Call It

- **Zhao & Meng (2014), arXiv 1405.2664**: Call their method **"FastMMD"** — described as an "ensemble of circular discrepancy" using Bochner's theorem and the Fourier transform. Never use "Fourier MMD."
- **Goudet et al. (2017), arXiv 1709.05321**: Refer to it as an **"approximate MMD criterion"** for scalability. No distinct name coined.

### Candidate Names

| Name | Accuracy | Discoverability | Disambiguation | Precedent |
|------|----------|-----------------|----------------|-----------|
| **Fourier MMD** | Medium | Poor (Graph FMMD dominates search) | Poor | Java upstream only |
| **FastMMD** | High | Good (unambiguous) | Good | Source paper (Zhao & Meng) |
| **RFF-MMD** | High | Good (academic) | Good | Recent papers (2025), one library |
| **ApproxMMD** | High | Good | Good | Goudet et al. usage ("approximate MMD") |

### What Other Libraries Use

- **alibi-detect**: `MMDDrift` (exact kernel, not RFF-based)
- **FlopsKa/rff-change-detection**: `StreamingRFFMMD` (RFF-based, closest match)
- **DrVAE**: `mmd_fourier` (internal function, not public-facing)
- **Recent literature** (arXiv 2505.17789, 2025): Consistently uses **"RFF-MMD"** as shorthand

### Java Upstream

The Java trustyai-explainability library uses `FourierMMD` as the class name. Files:
- `org.kie.trustyai.metrics.drift.fouriermmd.FourierMMD`
- `org.kie.trustyai.metrics.drift.fouriermmd.FourierMMDFitting`

---

## 2. Algorithm Overview

The metric approximates the Maximum Mean Discrepancy (MMD) between two distributions using Random Fourier Features (RFF). This avoids the O(n^2) cost of exact kernel MMD by projecting data into a low-dimensional feature space where inner products approximate kernel evaluations (Rahimi & Recht, 2007).

### Training Phase (`precompute` / `learn`)

1. **Preprocess**: Extract numeric columns. If `delta_stat=True`, compute first differences: `x = data[1:] - data[:-1]`
2. **Compute scale**: `scale = std(x, axis=0) * sig`, floored at `epsilon` to avoid division by zero
3. **Generate random features** (seeded for reproducibility):
   - `wave_num ~ N(0, 1)`, shape `(n_features, n_mode)`
   - `bias ~ U(0, 2*pi)`, shape `(1, n_mode)`
4. **Sample reference data**: Up to `n_window * n_test` samples without replacement
5. **Compute reference Fourier coefficients**: `a_ref = mean(cos(X/scale @ wave_num + bias)) * sqrt(2/n_mode)`
6. **Estimate null distribution**: For `n_test` iterations, sample a random window of size `n_window`, compute its Fourier coefficients, measure L2 distance to `a_ref`. Store `mean_mmd` and `std_mmd` of these distances.

### Inference Phase (`calculate` / `execute`)

1. **Preprocess** test data identically (delta if needed)
2. **Regenerate** same `wave_num` and `bias` using same seed
3. **Compute test Fourier coefficients** using stored scale
4. **MMD**: `mmd = sum((a_ref - a_test)^2)` (squared L2 distance)
5. **Normalize**: `drift_score = max((mmd - mean_mmd) / std_mmd, 0)`
6. **P-value**: `p_value = 1 - Phi(gamma - drift_score)` where Phi is the standard normal CDF
7. **Decision**: `drift_detected = (p_value > threshold)`

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_window` | 168 | Samples per MMD window computation |
| `n_test` | 100 | Number of MMD scores computed during training |
| `n_mode` | 512 | Number of random Fourier modes (features) |
| `sig` | 10.0 | Kernel scale multiplier (applied to per-feature std) |
| `delta_stat` | True | Compute on first differences instead of raw data |
| `epsilon` | 1e-7 | Floor for scale to prevent division by zero |
| `gamma` | 2.0 (Java) / 1.5 (Python) | Drift threshold offset in normal CDF |
| `threshold` | 0.8 (Python) | P-value threshold for flagging drift |
| `seed` | 1234 (Python) / 22 (Java) | Random seed for reproducibility |

### Key Mathematical Identity

The approximation relies on Bochner's theorem: for a shift-invariant kernel `k(x-y)`, there exists a spectral measure such that:

```
k(x - y) = E_w[cos(w^T x + b) * cos(w^T y + b)]
```

By sampling `n_mode` random frequencies `w` from the spectral distribution (standard normal for RBF kernel) and random phases `b` from `U(0, 2*pi)`, the mean of `cos(w^T x + b) * sqrt(2/n_mode)` gives a finite-dimensional feature map whose inner product approximates the kernel.

---

## 3. Existing Implementations

### Java (trustyai-explainability)

**Files:**
- Core: `explainability-core/.../metrics/drift/fouriermmd/FourierMMD.java`
- Fitting: `explainability-core/.../metrics/drift/fouriermmd/FourierMMDFitting.java`
- Endpoint: `explainability-service/.../endpoints/metrics/drift/FourierMMDEndpoint.java`
- Request: `explainability-service/.../payloads/metrics/drift/fouriermmd/FourierMMDMetricRequest.java`
- Parameters: `explainability-service/.../payloads/metrics/drift/fouriermmd/FourierMMDParameters.java`
- Tests: `explainability-core/.../metrics/drift/fouriermmd/FourierMMDTest.java`

**Key classes:**
- `FourierMMD`: Main class with `precompute()` (static, returns fitting) and `calculate()` (instance, returns `HypothesisTestResult`)
- `FourierMMDFitting`: Encapsulates trained state (`scale[]`, `aRef[]`, `meanMMD`, `stdMMD`, `randomSeed`, `nMode`, `deltaStat`)
- Returns `HypothesisTestResult(statVal, pValue, reject)`

### Python Reference (AI-TS-Drift)

**File:** `aitsdrift/core/data_drift/distance.py`

**Classes:**
- `Fourier_MMD`: Main implementation with `learn()` and `execute()` methods
- `RBF_MMD`: Exact kernel MMD (separate class, not RFF-based)

**Notable differences from Java:**
- Uses `np.random.seed()` for reproducibility (Java uses seeded `Random` object)
- P-value formula: `1 - norm.cdf(-score + gamma)` vs Java's `1 - norm.cdf(gamma - drift_score)` (algebraically equivalent)
- Default `gamma`: 1.5 (Python) vs 2.0 (Java)
- Default `seed`: 1234 (Python) vs 22 (Java)

---

## 4. TrustyAI Service Patterns (for Implementation)

### Architecture

The service follows a three-layer pattern:

1. **Core** (`src/core/metrics/drift/`): Pure algorithm, static `calculate()` method, no HTTP/storage dependencies. Returns `dict[str, float | bool]`.
2. **Endpoint** (`src/endpoints/metrics/drift/`): FastAPI router with 5 endpoints (compute, definition, schedule, delete, list). Uses Pydantic request models with `Field(alias=...)` for camelCase API compatibility.
3. **Tests**: Factory-based (`tests/core/metrics/drift/factory.py`) for standard scenarios + custom tests for metric-specific behavior.

### Canonical Template

The **CompareMeans** endpoint (`src/endpoints/metrics/drift/compare_means.py`) is the canonical template. Key patterns:
- `METRIC_NAME` constant
- `BaseMetricRequest` subclass with `@model_validator` for default metric name
- Per-feature iteration over `fit_columns`
- Aggregation of per-feature results
- Standard response: `{"status", "value", "drift_detected", ..., "feature_results"}`
- 5 endpoints: POST compute, GET definition, POST schedule, DELETE schedule, GET list

### Registration

New metrics are registered by importing and including the router in `src/main.py`.

---

## 5. Design Considerations for This Metric

### Multivariate Nature

Unlike CompareMeans or KS (which operate per-feature), this MMD metric is inherently **multivariate** — it computes a single statistic across all features simultaneously. The Fourier features are computed on the full feature matrix, not per-column.

This affects the endpoint pattern:
- `fit_columns` specifies which columns to include in the multivariate computation
- Result is a single drift score, not per-feature results
- No `feature_results` dict needed (or it could contain a single entry)

### Statefulness

The metric has a **fitting/training phase** that produces parameters (`a_ref`, `mean_mmd`, `std_mmd`, `scale`) needed at inference time. This is different from stateless metrics like CompareMeans where each computation is independent.

Options:
- Compute fitting from reference data on every request (simpler, but slower)
- Cache fittings keyed by (model_id, reference_tag, parameters) (faster, more complex)
- Store fitting as part of the scheduled request (needed for Prometheus scheduling)

### Seed Reproducibility

The same random seed must produce the same `wave_num` and `bias` at both training and inference time. NumPy's `np.random.default_rng(seed)` (preferred over legacy `np.random.seed()`) ensures this.

---

## 6. Modern Alternatives (2020--2026 Literature Survey)

FastMMD (2014) is now over a decade old. The field has advanced significantly. Below is a survey of methods that improve upon or supersede it.

### 6.1 Known Weaknesses of FastMMD / RFF-MMD

1. **Pointwise inconsistency with finite features.** The RFF-MMD test is consistent only as the number of random features L approaches infinity. With a fixed L (e.g., 512), the test can miss genuine distributional differences.
2. **Cubic-time worst case.** Matching the power of the exact quadratic-time MMD test with RFF requires Omega(n^3) computation — paradoxically *worse* than exact MMD (Domingo-Enrich et al., 2023).
3. **Bandwidth sensitivity.** The median heuristic for kernel bandwidth selection can be highly suboptimal. FastMMD's fixed `sig` parameter provides no adaptive bandwidth selection.

### 6.2 Theoretical Advances for RFF-MMD

**Computational-Statistical Trade-off in Kernel Two-Sample Testing with RFF** — Choi & Kim, 2024 ([arXiv:2407.08976](https://arxiv.org/abs/2407.08976)). First rigorous characterization of the power-computation trade-off. Key finding: the number of random features L should scale as O(n^(2s/(2s+d))) for Sobolev ball alternatives of smoothness s to achieve minimax separation rates. A fixed L=512 is *not* theoretically justified — it should grow with sample size.

**Minimax Optimal Kernel Two-Sample Tests with Random Features** — Mukherjee et al., 2025 ([arXiv:2502.20755](https://arxiv.org/abs/2502.20755)). Proposes a spectral-regularized RFF test achieving minimax optimality for a broader class of alternatives. Tightest known analysis of RFF-based testing.

### 6.3 Better Batch Two-Sample Tests

#### Compress Then Test (CTT) — Domingo-Enrich, Dwivedi & Mackey, AISTATS 2023

[arXiv:2301.05974](https://arxiv.org/abs/2301.05974)

**The strongest competitor to FastMMD.** CTT compresses each n-point sample into a small coreset using KT-Compress, then runs the exact quadratic MMD on the compressed samples. Key advantages:

- Near-linear time with **no approximation error** (unlike RFF which introduces bias)
- Inherits the same optimal detection boundary as the full quadratic-time test
- Empirically provides **20--200x speedups** over approximate MMD tests with no power loss
- Compression is done once regardless of permutation count
- Implementation: Microsoft [`goodpoints`](https://github.com/microsoft/goodpoints) Python package

#### Cross-MMD — Shekhar, Kim & Ramdas, NeurIPS 2022

[arXiv:2211.14908](https://arxiv.org/abs/2211.14908)

Uses sample-splitting and studentization to produce a test statistic with a **limiting standard Gaussian distribution** under the null. Eliminates the need for permutation testing entirely (closed-form threshold). Minimax rate-optimal with Gaussian kernel. Can be combined with CTT compression.

#### Block MMD (B-test) — Zaremba, Gretton & Blaschko, NeurIPS 2013

Partitions kernel matrices into blocks of size B, computes MMD per block, and averages. With B = sqrt(n), achieves O(d*n^1.5). Higher power than linear-time MMD but largely superseded by CTT.

#### euMMD — Bodenham & Kawahara, Statistics and Computing 2023

[Springer link](https://link.springer.com/article/10.1007/s11222-023-10271-x). Exact MMD for univariate data in O(n log n) via Laplacian kernel and sorting. Loses exactness in higher dimensions.

### 6.4 Bandwidth / Kernel Selection

#### MMDAgg — Schrab, Kim, Albert, Laurent, Guedj & Gretton, JMLR 2023

[arXiv:2110.15073](https://arxiv.org/abs/2110.15073)

Solves the kernel bandwidth selection problem. Aggregates MMD tests across multiple bandwidths and kernel families (Gaussian + Laplace with ~10 bandwidths each) using a multiple testing correction. Properties:

- Achieves minimax rate over Sobolev balls
- Non-asymptotic Type I error control
- No held-out data needed for kernel selection
- Has a linear-time variant using incomplete U-statistics
- Code: [github.com/antoninschrab/mmdagg](https://github.com/antoninschrab/mmdagg) (NumPy + JAX, ~100x faster with JAX)

**This directly addresses FastMMD's biggest practical weakness** — the sensitivity to the fixed `sig` parameter.

#### Deep Kernel MMD — Liu et al., ICML 2020

Trains a deep kernel k(x,y) = (1-eps)*k_a(phi(x), phi(y)) + eps*k_b(x,y) where phi is a learned projection, optimizing the MMD-to-variance ratio. Dramatically more powerful for complex structured data (images, embeddings). Integrated into alibi-detect. Quadratic cost + training overhead, but highest power for subtle shifts.

### 6.5 Online / Streaming Change Detection

#### Online RFF-MMD — Kalinke & Gavioli-Akilagun, 2025

[arXiv:2505.17789](https://arxiv.org/abs/2505.17789)

The current **state-of-the-art for online kernel-based change detection**. Uses RFF-approximated MMD embedded in a dyadic grid sequential testing scheme.

Key properties:
- **O(r log n)** time and space per observation (r = number of RFFs)
- **No window parameter** — the dyadic grid handles all scales automatically
- **No reference/training data** from the pre-change distribution needed
- **Minimax optimal** detection delay up to log factors (first such result for kernel methods)
- **Formal false alarm guarantees** — ARL control or uniform false alarm probability
- Multiple change point extension
- Code: [github.com/FlopsKa/rff-change-detection](https://github.com/FlopsKa/rff-change-detection)

**Comparison to FastMMD for streaming:** Online RFF-MMD is a substantial improvement. FastMMD requires a fixed window, reference data, and has no formal false alarm control. However, for **batch** two-sample testing (compare reference dataset to current dataset), the sequential machinery is unnecessary overhead.

Default parameters from the paper:
- Kernel: Gaussian with **median heuristic** bandwidth
- Number of RFFs: r = 1000 in experiments
- Threshold calibration: Monte Carlo (10x target ARL samples under null)

### 6.6 What Modern Libraries Use

| Library | MMD Method | Kernel Selection | Complexity |
|---------|-----------|-----------------|------------|
| **alibi-detect** | Exact quadratic MMD + permutation test; KeOps for scaling | Median heuristic or learned deep kernel | O(n^2) |
| **Frouros** | Exact MMD (Gretton 2012) + permutation test | RBF with configurable bandwidth | O(n^2) |
| **Evidently AI** | MMD for embedding drift | Default settings | O(n^2) |
| **TorchDrift** | MMD + bootstrapping | Gaussian kernel | O(n^2) |
| **River** | No MMD (uses DDM, ADWIN for streaming) | N/A | N/A |

**No major library currently implements RFF-based / FastMMD-style approximate MMD.** They all use exact quadratic MMD with permutation testing, relying on GPU acceleration (KeOps) or sample-size limiting.

### 6.7 Summary and Recommendation

| Method | Time | Power | Bandwidth-free? | Online? | Code Available? |
|--------|------|-------|-----------------|---------|-----------------|
| FastMMD (RFF, 2014) | O(Lnd) | Moderate (finite L) | No | No | Java upstream |
| B-test (block, 2013) | O(d*n^1.5) | High | No | No | Yes |
| MMDAgg (2023) | O(n^2 * B) | Optimal (adaptive) | **Yes** | No | [Yes](https://github.com/antoninschrab/mmdagg) |
| CTT (coresets, 2023) | Near-linear | **Optimal** | No | No | [Yes](https://github.com/microsoft/goodpoints) |
| Cross-MMD (2022) | O(n^2) | Optimal | No | No | Yes |
| Online RFF-MMD (2025) | O(r log n) per step | Optimal | No | **Yes** | [Yes](https://github.com/FlopsKa/rff-change-detection) |

**For batch drift detection (our use case):**

1. **Best overall:** CTT (Compress Then Test) — near-linear time, no approximation error, optimal power, production-ready code. However, it has a dependency on the `goodpoints` package (MIT license, Microsoft).
2. **Best for robust kernel selection:** MMDAgg — eliminates bandwidth sensitivity, the biggest practical weakness of FastMMD. Can be combined with CTT.
3. **Pragmatic minimum:** Implement FastMMD as specified (for Java parity), but consider adding MMDAgg-style bandwidth aggregation as a future enhancement.

**For streaming drift detection (future work):**
- Online RFF-MMD (Kalinke 2025) is the clear winner.

---

## 8. `goodpoints` Package Evaluation (2026-05-13)

### Maintenance Concerns

The [`goodpoints`](https://github.com/microsoft/goodpoints) package (v0.3.2 on PyPI, v0.4.0 on main) has several red flags:

- **No unit tests** in the repository
- **Infrequent, narrow updates** — recent commits are limited to adding individual parameters (`skip_swap`), not maintenance or compatibility work
- **NumPy 2.x incompatibility** — Cython extensions fail with NumPy >= 2.0 (our project uses 2.4.4)
- **Python 3.12+ build failure** — `setup.py` uses `distutils` (removed in Python 3.12)
- **Zero open issues** — suggests low community engagement, not zero bugs

### NumPy 2.x Incompatibility Root Cause

The package has 5 Cython extension modules: `cttc.pyx`, `ktc.pyx`, `gaussianc.pyx`, `compressc.pyx`, `sobolevc.pyx`.

**What works:** `cttc.pyx` (8 KB) — uses only typed memoryviews and `libc.math`. The `ctt.rff()` function (RFF-MMD permutation test) depends solely on this module and works fine.

**What breaks:** `ktc.pyx` (29 KB) imports NumPy's **internal, unstable random C API**:

```python
from numpy.random cimport bitgen_t
from numpy.random.c_distributions cimport (random_standard_uniform,
      random_standard_uniform_fill)
```

These APIs changed in NumPy 2.0. The `ctt.ctt()` function (the actual Compress Then Test algorithm) calls `compress.compress_kt()` → `ktc.thin_K()`, which triggers the breakage. The `gaussianc.pyx` (49 KB) and `compressc.pyx` (18 KB) modules have similar issues.

**Estimated fix effort upstream:** 2--4 days for someone comfortable with Cython and NumPy C internals. See `goodpoints-numpy2-compat.md` for the full analysis.

### Alternative Libraries Evaluated

| Library | MMD Method | Dependencies | Maintenance | Verdict |
|---------|-----------|--------------|-------------|---------|
| [alibi-detect](https://github.com/SeldonIO/alibi-detect) | Exact quadratic + permutation | **Requires TensorFlow or PyTorch** | Active (v0.13.0, Dec 2025) | Too heavy |
| [TorchDrift](https://torchdrift.org/) | MMD + bootstrapping | PyTorch | Moderate | Too heavy |
| [mmdagg](https://github.com/antoninschrab/mmdagg) | Aggregated MMD | NumPy + SciPy | Academic, low activity | Interesting but niche |
| [kernel_two_sample_test](https://github.com/emanuele/kernel_two_sample_test) | Exact MMD (Gretton 2012) | NumPy + sklearn | Unmaintained | Reference only |

**No major library provides a lightweight, dependency-free MMD implementation suitable for production use.** All either require deep learning frameworks or are unmaintained academic code.

### Decision: Implement Our Own

The MMD two-sample test with permutation-based p-value is a well-understood algorithm (Gretton et al., JMLR 2012). The exact quadratic-time implementation is approximately 60 lines of pure NumPy:

1. Compute Gaussian RBF kernel matrix: `K[i,j] = exp(-||x_i - x_j||² / 2σ²)`
2. MMD² U-statistic: `sum(Kxx)/(n(n-1)) + sum(Kyy)/(m(m-1)) - 2*sum(Kxy)/(nm)`
3. Permutation test: shuffle labels B times, recompute statistic, p-value = `(#{null ≥ observed} + 1) / (B + 1)`

**Advantages of our own implementation:**

- **Zero new dependencies** — pure NumPy (already a project dependency)
- **Optimal statistical power** — exact MMD, no RFF approximation error
- **Fast for our batch sizes** — O(n²) for n=100--1000 is milliseconds
- **Clean upstream migration** — fits perfectly in `src/core/` → `trustyai-explainability-python`
- **Full test coverage** — existing test suite validates statistical behavior, not implementation internals

**What about CTT compression?** The "Compress" step of CTT (kernel thinning) is only beneficial for large samples (n > 5000) where O(n²) becomes expensive. For our monitoring batch sizes (100--1000), exact MMD is both faster and simpler than compressing first. Kernel thinning can be added as a future optimization behind the same `MMD.compute()` interface if large-sample support is needed.

---

## 9. References

1. Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, Alexander Smola, "A Kernel Two-Sample Test," JMLR 13, 2012.
2. Ji Zhao, Deyu Meng, "FastMMD: Ensemble of Circular Discrepancy for Efficient Two-Sample Test," arXiv:1405.2664, 2014.
2. Olivier Goudet, et al., "Learning Functional Causal Models with Generative Neural Networks," arXiv:1709.05321, 2017.
3. Ali Rahimi, Benjamin Recht, "Random Features for Large-Scale Kernel Machines," NIPS, 2007.
4. Leone et al., "Graph Fourier MMD for Signals on Graphs," 2023 (unrelated — this is what "Fourier MMD" search returns).
5. Choi & Kim, "Computational-Statistical Trade-off in Kernel Two-Sample Testing with RFF," arXiv:2407.08976, 2024.
6. Mukherjee et al., "Minimax Optimal Kernel Two-Sample Tests with Random Features," arXiv:2502.20755, 2025.
7. Domingo-Enrich, Dwivedi & Mackey, "Compress Then Test: Powerful Kernel Testing in Near-Linear Time," arXiv:2301.05974, AISTATS 2023.
8. Shekhar, Kim & Ramdas, "Permutation-Free Kernel Two-Sample Testing," arXiv:2211.14908, NeurIPS 2022.
9. Schrab, Kim, Albert, Laurent, Guedj & Gretton, "MMD Aggregated Two-Sample Test," arXiv:2110.15073, JMLR 2023.
10. Liu et al., "Learning Deep Kernels for Non-Parametric Two-Sample Tests," ICML 2020.
11. Kalinke & Gavioli-Akilagun, "Optimal Online Change Detection via Random Fourier Features," arXiv:2505.17789, 2025.
12. Zaremba, Gretton & Blaschko, "B-test: A Non-parametric, Low Variance Kernel Two-Sample Test," NeurIPS 2013.
13. Bodenham & Kawahara, "euMMD: Efficiently Computing the MMD in O(N log N) Time," Statistics and Computing, 2023.
