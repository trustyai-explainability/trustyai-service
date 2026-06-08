# JAX Evaluation for MMD Drift Detection

**Date:** 2026-05-14
**Decision:** Do not adopt

## What is JAX?

Google's library for high-performance numerical computing. Key features:

- **XLA compilation** — JIT-compiles NumPy-like code to optimized machine code (CPU, GPU, TPU)
- **Automatic differentiation** — not relevant for MMD (no gradients needed)
- **NumPy-compatible API** — `jax.numpy` is a near drop-in for `numpy`
- **Vectorization** — `jax.vmap` auto-vectorizes over batch dimensions, eliminating Python loops

## Where JAX Would Help

The MMD bottleneck is kernel matrix computation (O(n²) pairwise distances) and permutation/bootstrap loops. JAX can accelerate both:

| Scenario | NumPy CPU | JAX CPU | JAX GPU |
|----------|-----------|---------|---------|
| MMDAgg (n=500, d=10) | 43s | 15s (~3x) | 0.5s (~90x) |

Source: Schrab et al., JMLR 2023, author benchmarks.

1. **Kernel matrices** — XLA fuses distance + kernel evaluation into a single compiled kernel
2. **Permutation loops** — `jax.vmap` over B permutations replaces the Python loop entirely
3. **Bandwidth grid** — multiple kernel evaluations vectorize naturally

## Why Not

### No GPU in production

TrustyAI runs on OpenShift pods without GPU access. The ~90x GPU speedup doesn't apply. On CPU alone, JAX gives ~3x from XLA compilation — meaningful for large workloads, but not for ours.

### Our batch sizes don't need it

Monitoring batch sizes are typically 100–1000 samples. CTT on CPU completes in <100ms for these sizes. There is no performance problem to solve.

### Massive dependency weight

| Dependency | Size |
|-----------|------|
| jaxlib (CPU) | ~200MB |
| jaxlib (CUDA/GPU) | ~1–2GB |
| Current MMD dependencies | 0 (beyond numpy/scipy) |

### JIT cold start

JAX's first call triggers JIT compilation: 2–5s overhead. For a service computing MMD every 30s on small batches, the JIT overhead dominates the actual computation time.

### Stale JAX implementations

- goodpoints' `goodpoints/jax/` directory pins `jaxlib=0.4.1` (Jan 2023), untested with current JAX
- Schrab's `mmdagg` JAX variant has the same staleness issue

### Complexity cost

JAX requires functional purity (no in-place mutation), explicit device placement, and different debugging patterns. Maintaining both NumPy and JAX code paths doubles the surface area.

## Effort Estimate

| Approach | Effort | Risk |
|----------|--------|------|
| Use goodpoints JAX variants | ~1 day | Stale code, may not work |
| JAX-ify our own code | 3–5 days | Must maintain dual paths |
| CI infrastructure | 1–2 days | Need JAX-capable runners |

## When to Revisit

- **n > 10,000 samples** — but CTT compression (already available) is the better scaling solution
- **GPU-equipped pods** — if the deployment model changes to include GPU nodes
- **Streaming MMD** — Online RFF-MMD (Kalinke 2025) might benefit from JAX for high-throughput streams
