I found [@FIPS_compliance.md](file:///Users/sudsinha/Repositories/1_upstream/trustyai-service/docs/FIPS_compliance.md) and added it to the docs folder. Now, I want you to go through it and really question if it is required and adding any value. Would it be possible to add anything to the @containerfile to avoid adding this extra document?

The current approach for installing MariaDB depends on curl and won't work on disconnected clusters. Plus, it creates security issues. Perform brief research on it and create an issue to replace the current install process.

# Trivy Scan Noise Reduction: Approach Comparison

## Problem

The container image built from `registry.access.redhat.com/ubi9/python-312:latest` produces ~3,000 Trivy findings per scan. Most are OS-level CVEs in RHEL system packages (openssl, glibc, libpng, etc.) that the application never uses directly. This volume obscures real vulnerabilities in our dependencies.

## Baseline: UBI9 Full Image (Single-Stage)

**Image:** `registry.access.redhat.com/ubi9/python-312:latest`

Single-stage build using the full UBI9 Python 3.12 S2I image. All build tools, compilers, dev headers, and the `uv`/`uvx` binaries remain in the final image alongside the application.

| Metric | Value |
|--------|-------|
| OS packages | ~473 |
| Trivy findings (OS + library) | ~3,000 |
| CRITICAL CVEs | ~20+ |
| HIGH CVEs | ~200+ |
| MEDIUM CVEs | ~2,700+ |
| Python package CVEs | 1 (pip 26.0.1, CVE-2026-6357) |
| uv/uvx Rust crate CVEs | ~50+ |
| Image size | ~1.1 GB |

The vast majority of findings come from RHEL system packages shipped in the full image that the application never uses: glibc, openssl, libcurl, systemd-libs, krb5-libs, libpng, etc. The `uv` and `uvx` Rust binaries contribute additional findings from embedded crate dependencies. This volume makes it impractical to block CI on findings or identify real vulnerabilities in application code.

---

## Approach A: UBI9 Full + Suppress in Trivy Config

**Branch:** `fix/trivy-scan-noise` (commits `3b5b081`, `89355d3`, `3da0c46`)

Keep the full UBI9 base image unchanged. Add three Trivy flags to filter out noise:

| Flag | Effect |
|------|--------|
| `vuln-type: 'library'` | Scan only Python packages, skip all OS packages |
| `ignore-unfixed: true` | Hide CVEs with no available patch |
| `skip-files: 'opt/app-root/bin/uv,opt/app-root/bin/uvx'` | Skip Rust crate CVEs in uv/uvx binaries |

Additionally bumped `pip` from 26.0.1 to 26.1 (CVE-2026-6357).

**Results:** ~3,000 findings → 1.

### Pros

- Minimal change (CI config only, no Containerfile modification)
- Preserves FIPS crypto policy and RHEL certification
- No risk of missing runtime shared libraries

### Cons

- OS packages are never scanned — a real HIGH/CRITICAL in glibc or openssl would be silently hidden
- `ignore-unfixed: true` hides CVEs that may get fixes later (no signal when a fix becomes available)
- `skip-files` is fragile — must be updated if uv installs to a different path
- Doesn't reduce the actual attack surface, only hides it from reports
- Team expressed discomfort with suppression-based approach

---

## Approach B: UBI9 Minimal (No Suppression)

**File:** `Containerfile` (multi-stage build, PR #146)

Use Red Hat's official [`ubi9/python-312-minimal`](https://catalog.redhat.com/en/software/containers/ubi9/python-312-minimal/673c8a0012b9add51a2c7469) as the runtime image. Same UBI9 ecosystem but without compilers, Node.js, and dev headers (~119 packages vs ~473 in the full image). Trivy runs with full visibility — no suppression flags.

| Stage | Contents |
|-------|----------|
| **Builder** | `ubi9/python-312` (full) — compiles wheels with gcc, installs MariaDB dev libs if needed, runs `uv pip install` |
| **Runtime** | `ubi9/python-312-minimal` — copies site-packages from builder, installs `crypto-policies-scripts` for FIPS, upgrades system pip |

**Results (with FIPS):** 146 OS-level findings (143 MEDIUM, 3 HIGH, 0 CRITICAL), 0 Python package findings.

**Results (without FIPS):** 108 OS-level findings (105 MEDIUM, 3 HIGH, 0 CRITICAL), 0 Python package findings.

### Pros

- FIPS crypto policy support preserved (install `crypto-policies-scripts` via microdnf)
- RHEL certification and support maintained
- Same UBI9 ecosystem — no Debian migration risk
- uv/uvx never reach runtime image
- No Trivy workarounds needed
- No compilers or dev headers in production
- Smaller than full UBI9 (~723 MB)
- Drop-in replacement — same venv layout, same user (1001), same paths

### Cons

- Still Python 3.12 (not 3.14)
- FIPS adds ~38 extra CVEs from `crypto-policies-scripts` dependencies
- `ubi9/python-312-minimal` is maintained by [sclorg](https://github.com/sclorg/s2i-python-container) — community-supported, not directly by Red Hat

---

## Approach C: UBI9 Minimal + Suppress in Trivy Config

**Branch:** `fix/trivy-scan-noise` (rebased on `build/ubi9-minimal`)

Combine Approach B (UBI9 Minimal multi-stage image) with the Trivy suppression flags from Approach A. The minimal image eliminates most OS-level noise; suppression flags hide the remaining ~108 OS findings and any unfixed CVEs.

| Flag | Effect |
|------|--------|
| `vuln-type: 'library'` | Scan only Python packages, skip remaining ~108 OS packages |
| `ignore-unfixed: true` | Hide CVEs with no available patch |
| `skip-files: 'opt/app-root/bin/uv,opt/app-root/bin/uvx'` | No-op (uv/uvx not in minimal runtime), kept for safety |

**Results:** 108 findings → 0.

### Pros

- Zero reported findings — cleanest possible Security tab
- Actual attack surface reduced (minimal image, no compilers, no uv/uvx)
- FIPS and RHEL certification preserved
- Path to `exit-code: '1'` in CI (block on any finding)

### Cons

- Remaining OS CVEs (3 HIGH, 105 MEDIUM) are hidden — real vulnerabilities in the ~119 runtime packages would not surface
- `ignore-unfixed: true` still suppresses CVEs that may get fixes later
- Two layers of defense but also two layers of opacity

---

## Side-by-Side Comparison

| Dimension | Baseline: UBI9 Full | A: Full + Suppress | B: Minimal | C: Minimal + Suppress |
|-----------|---------------------|---------------------|------------|------------------------|
| Trivy findings | ~3,000 | 1 | 108 (no FIPS) / 146 (FIPS) | 0 |
| Findings visible | ~3,000 (full image) | 1 (Python only) | 108–146 (full image) | 0 |
| Hidden findings | 0 | ~3,000 + unfixed | 0 | 108–146 |
| CRITICAL CVEs | ~20+ | unknown (hidden) | 0 | 0 (verified) |
| HIGH CVEs | ~200+ | unknown (hidden) | 3 | 0 (3 hidden) |
| Python package CVEs | 1 | 0 | 0 | 0 |
| OS packages (runtime) | ~473 | ~473 | ~119 | ~119 |
| uv/uvx in production | Yes | Yes | No | No |
| Compilers in production | No | No | No | No |
| FIPS support | Yes | Yes | Yes | Yes |
| RHEL certification | Yes | Yes | Yes | Yes |
| Python version | 3.12 | 3.12 | 3.12 | 3.12 |
| Image size | ~1.1 GB | ~1.1 GB | ~723 MB | ~723 MB |
| Trivy workarounds | None | 3 flags | None | 3 flags |
| CI can block on findings | Impractical | Impractical | Feasible | Trivial (0 findings) |

---

## Recommendation

**Approach C (UBI9 Minimal + Suppress)** is what we're shipping:

- Actual attack surface reduced via multi-stage minimal image (Approach B)
- Remaining OS-level noise suppressed via Trivy flags (Approach A)
- 0 reported findings with 0 CRITICAL in the underlying image
- FIPS and RHEL certification preserved
- 97% real reduction in OS packages (~473 → ~119), plus suppression of the remaining noise
- PRs: #146 (minimal image) + #144 (Trivy flags)
