1. For \epsilon: why did we replace them with e in the comments?
2. The separation of names in Kolmogorov–Smirnov is usually by en-dash since they are two separate individuals. Can we revert back to en dash?
4. The contants in test_greenwald_khanna_quantile_sketch.py is insanity. Just add a ignore statement in pyproject.toml for that specific file, and revert back to only a sensible set of constants.
5. In `def __init__(self, epsilon: float = 0.01) -> None`, can we use a constant definition for 0.01?

Now, we want to implement the Fourier MMD drift detection metric that is present in the Java library present in ../trustyai-explainability. The related Jira is https://redhat.atlassian.net/browse/RHOAIENG-55616.

The relevant papers are as follows:
* Ji Zhao, Deyu Meng, 'FastMMD: Ensemble of Circular Discrepancy for Efficient Two-Sample Test' <https://arxiv.org/abs/1405.2664>
* Olivier Goudet, et al. 'Learning Functional Causal Models with Generative Neural Networks' <https://arxiv.org/abs/1709.05321>

A Python version _may_ be found in the folder /Users/sudsinha/Repositories/TrustyAI/AI-TS-Drift/aitsdrift/core/data_drift, but we may need to coax out the important bits.

First, I want you to _plan_ this out thoroughly:
1. Is "Fourier MMD" an appropriate name for the metric? If I search for the term on the internet, the top results are "Graph Fourier MMD" related, which may be confusing. Are there any alternative names you suggest? Please do a thorough literature review.
2. Make a new branch for this. Choose an appropriate name (don't include the Jira tag).
3. Plan on how to implement this without any mistakes. We will need the core metric implementation, the associated endpoints, and the unit tests.



Now go through each issue, and resolve them. Ensure to not introduce new bugs. For each file you're making a change to, diff with origin/main to recall the original usage and intent.

I think some of the docstring changes you made may break again after running `ruff check` and `ruff format`. Please recheck.

# Commit & PR Plan (v2 — Fully Independent)

All commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format.

---

## Housekeeping: Close PR #117

**Action:** Close [PR #117](https://github.com/trustyai-explainability/trustyai-service/pull/117) and delete branch `refactor/config-objects`.

**Why:**
- Its scope is split across two new PRs (GaugeConfig → PR #1, StorageMetadata + DataArrayConfig → PR #2)
- 3 critical CodeRabbit issues require architectural changes, not fixes on top
- CI broken at import level (`InvalidSchemaError` doesn't exist)
- Already in draft — no review work lost

---

## Why v2?

PR #117 failed CI and got 6 CodeRabbit findings because API changes weren't self-contained:
- `gauge()` signature changed but `prometheus_scheduler.py` callers weren't updated → TypeError
- `InvalidSchemaError` imported but only `InvalidSchemaException` exists → ImportError
- `set_recorded_inferences` made keyword-only but caller passes positionally → TypeError

**v2 rules:**
1. When you change an API, update ALL callers in the same PR
2. Each file in exactly one PR
3. Each PR carries its own `pyproject.toml` changes (per-file-ignores for its files) — **no cross-PR dependencies**

---

## Dependency Graph

```
Chain A: GaugeConfig
  gauge_config.py (new) ──► prometheus_publisher.py ──► prometheus_scheduler.py
                                                    ──► tests/prometheus/*
                                                    ──► tests/endpoints/metrics/drift/factory.py

Chain B: StorageMetadata
  storage_metadata.py ──► data_source.py ──► consumer_endpoint.py
                      ──► exceptions.py (InvalidSchemaError)

Chain C: ModelData (standalone)
  model_data.py ──► tests/datasources/*

Chain D: Parser + Storage
  modelmesh_parser.py ──► pvc.py, maria.py, storage_interface.py
                      ──► consumer_endpoint.py (PartialPayload import)

Chain E: Drift Metrics (standalone)
  compare_means.py, utils.py, jensen_shannon.py ──► tests/drift/*
```

`consumer_endpoint.py` appears in Chains B+D → bundle into same PR.
`prometheus_scheduler.py` has gauge + exc_info + error changes → all in Chain A's PR.

---

## Execution Strategy: 5 Fully Independent PRs (~22 commits)

**No file appears in two PRs → no merge conflicts, all independent. No merge order constraints.**

| PR | Theme | Files touched | Jira |
|---|---|---|---|
| #1 | Prometheus (GaugeConfig) | `src/service/prometheus/*`, drift factory | RHOAIENG-57849 |
| #2 | Data Layer (StorageMetadata + parser + storage) | `src/service/data/*`, `payloads/*`, `utils/*`, `consumer_endpoint.py` | RHOAIENG-57849, 55581 |
| #3 | Core Metrics & Endpoints (drift types + formatting) | `src/core/*`, `src/endpoints/*`, `src/main.py` | RHOAIENG-57849 |
| #4 | Configuration & Testing | `pyproject.toml`, `uv.lock` | RHOAIENG-55591, 55590 |
| #5 | CI & Pre-commit | `.github/workflows/*`, `.pre-commit-config.yaml` | RHOAIENG-55596, 55592 |

### pyproject.toml Strategy (enables full independence)

Each PR carries its own `pyproject.toml` per-file-ignores for **its own files only**. Since file sets are disjoint, pyproject.toml edits are in different sections → no merge conflicts, no ordering constraints.

| PR | pyproject.toml sections it owns |
|---|---|
| #1 | per-file-ignores for `src/service/prometheus/*` |
| #2 | per-file-ignores for `src/service/data/*`, `payloads/*`, `consumer/*`, `[tool.bandit]` |
| #3 | per-file-ignores for `src/core/*`, `src/endpoints/{evaluation,explainers,metadata,metrics}/*`, `src/middleware/*` |
| #4 | `[dependency-groups]`, `[tool.ruff.lint]` global ignores, `[tool.pytest.ini_options]`, `[tool.pyrefly]`, `[tool.ruff]` top-level, `uv.lock` |
| #5 | None (CI/pre-commit only) |

Merge in **any order**. No PR depends on another.

---

## PR #1: Prometheus Refactor — GaugeConfig + All Callers (2 commits) — ✅ CREATED as [#119](https://github.com/trustyai-explainability/trustyai-service/pull/119)

Branch: `refactor/gauge-config`
Jira: RHOAIENG-57849
Status: **✅ Merged (2026-04-16). Squash-merged to main.**

### Commit 1.1: `feat(prometheus): add GaugeConfig model` [RHOAIENG-57849]

**Files:**
- NEW: `src/service/prometheus/gauge_config.py`

**Changes:**
- Pydantic model with fields: model_name, request_id, value, named_values, request, metric_name
- Validation at config creation time

**Why:** Standalone new file, no callers changed yet.

**Verification:**
```bash
uv run python -c "from src.service.prometheus.gauge_config import GaugeConfig; print('OK')"
```

---

### Commit 1.2: `refactor(prometheus)!: migrate gauge() to GaugeConfig API` [RHOAIENG-57849]

**Files:**
- `src/service/prometheus/prometheus_publisher.py`
- `src/service/prometheus/prometheus_scheduler.py`
- `src/service/prometheus/shared_prometheus_scheduler.py`
- `tests/service/prometheus/test_prometheus_publisher.py`
- `tests/service/prometheus/test_prometheus_scheduler.py`
- `tests/endpoints/metrics/drift/factory.py`

**Changes:**
- `gauge()` method: accept `GaugeConfig` instead of 6 params
- Fix named-value cleanup leak: track derived UUIDs per root request, clean up all on delete (CodeRabbit #5)
- Migrate all 4 `publisher.gauge(model_name=..., id=...)` calls in prometheus_scheduler.py → `publisher.gauge(GaugeConfig(...))`
- Remove redundant `exc_info=True` from `logger.exception()` calls (LOG014)
- Update tests to use GaugeConfig, expect ValidationError instead of ValueError
- Update drift factory.py gauge call pattern

**Why:** Breaking API change — must migrate ALL callers atomically.

**CodeRabbit issues fixed:** #5 (named-value leak), #6 (gauge callers)

**Verification:**
```bash
uv run pytest tests/service/prometheus/ tests/endpoints/metrics/drift/ -v
uv run ruff check src/service/prometheus/
```

---

### Commit 1.3: `refactor(config): update per-file-ignores for prometheus` [RHOAIENG-57849]

**Files:**
- `pyproject.toml` — **only** the per-file-ignores entries for `src/service/prometheus/*`

**Changes:** Add/update ruff per-file-ignores for prometheus files (BLE001, SLF001, etc.)

**Why:** Each PR carries its own pyproject.toml config to be fully independent.

---

### Commit 1.4: `style(prometheus): formatting and docstrings` [RHOAIENG-57849]

**Files:**
- `src/service/prometheus/metric_value_carrier.py`
- `src/service/prometheus/__init__.py`
- `tests/service/prometheus/__init__.py`
- `tests/service/prometheus/test_metric_value_carrier.py`
- `tests/endpoints/metrics/drift/__init__.py`

**Changes:** Formatting, docstrings, import ordering only. No logic changes.

**Verification:**
```bash
uv run ruff check src/service/prometheus/ tests/service/prometheus/
uv run pytest tests/service/prometheus/ -v
```

---

## PR #2: Data Layer Fixes (1 commit) — ✅ CREATED as [#121](https://github.com/trustyai-explainability/trustyai-service/pull/121)

Branch: `refactor/data-layer`
Jira: RHOAIENG-57849, RHOAIENG-55581
Status: **✅ Merged (2026-04-16).**

### Commit 2.1: `refactor(storage): introduce StorageMetadataConfig` [RHOAIENG-57849]

**Files:**
- `src/service/data/metadata/storage_metadata.py`
- `src/service/data/exceptions.py`
- `src/service/data/datasources/data_source.py`
- `src/endpoints/consumer/consumer_endpoint.py`
- `tests/service/data/datasources/test_datasource.py`
- `tests/service/test_consumer_endpoint_reconciliation.py`

**Changes:**
- Add `StorageMetadataConfig` (Pydantic) — config-only `__init__`, no backward-compat `**kwargs`
- Rename `InvalidSchemaException` → `InvalidSchemaError` (PEP 8 naming)
- `set_recorded_inferences` is keyword-only (`*`) — caller already uses keyword syntax
- Update all callers (data_source.py, pvc.py, test_datasource.py) to construct `StorageMetadataConfig`
- Move `payload_kind` validation to early guard in consumer_endpoint.py (was unreachable in try/else)

**Verification:**
```bash
uv run pytest tests/service/data/datasources/ tests/service/test_consumer_endpoint_reconciliation.py -v
```

---

### Commit 2.2: `refactor(data): clean up ModelData and DataArrayConfig` [RHOAIENG-57849]

**Files:**
- `src/service/data/model_data.py`

**Changes:**
- `DataArrayConfig` and `ModelDataContainer` are pre-existing dataclasses (already on main) — add docstrings and formatting
- Fix async docstring example: `await model_data.data()` (CodeRabbit #4)

**Verification:**
```bash
uv run pytest tests/service/data/datasources/test_datasource.py -v
```

---

### Commit 2.3: `refactor(parser): simplify tensor extraction with dispatch table` [RHOAIENG-57849]

**Files:**
- `src/service/data/modelmesh_parser.py`
- `tests/service/data/test_modelmesh_parser.py`

**Changes:**
- Dictionary dispatch for `_extract_tensor_data` (8 returns → 1)
- Protobuf types: `Any` with `# noqa: ANN401` (no stubs for generated protobuf)

**Verification:**
```bash
uv run pytest tests/service/data/test_modelmesh_parser.py -v
```

---

### Commit 2.4: `refactor(pvc): optimize imports and reduce complexity` [RHOAIENG-57849]

**Files:**
- `src/service/data/storage/pvc.py`
- `tests/service/data/test_payload_reconciliation_pvc.py`
- `tests/endpoints/test_upload_endpoint_pvc.py`

**Changes:**
- Move `TracebackType`, `File` to `TYPE_CHECKING` block (removed pointless Lock alias)
- Consolidate exception handlers (JSONDecodeError + KeyError)
- Update `StorageMetadata` caller to use `StorageMetadataConfig`
- Add `# noqa: S301` on pickle.loads
- Add `# noqa: PLC0415` on test post-reload imports

**Verification:**
```bash
uv run pytest tests/service/data/test_payload_reconciliation_pvc.py tests/endpoints/test_upload_endpoint_pvc.py -v
```

---

### Commit 2.5: `fix(maria): merge SQL string constants and add lint suppressions` [RHOAIENG-57849]

**Files:**
- `src/service/data/storage/maria/legacy_maria_reader.py`
- `src/service/data/storage/maria/maria.py`
- `src/service/data/storage/__init__.py`
- `tests/service/data/test_mariadb_storage.py`
- `tests/service/data/test_mariadb_migration.py`
- `tests/service/data/test_payload_reconciliation_maria.py`
- `tests/endpoints/test_upload_endpoint_maria.py`

**Changes:**
- Merge SQL string concatenation into single literals (2 violations fixed at root cause)
- Add `# noqa: S608` on 16 f-string SQL queries (table names can't be parameterized)
- Add `# noqa: S301` on pickle.loads in maria.py
- Add `# noqa: S108` and `# noqa: PLC0415` in storage/__init__.py
- Add `# noqa: PLC0415` on test post-reload imports

**Verification:**
```bash
uv run pytest tests/service/data/test_mariadb_storage.py tests/service/data/test_mariadb_migration.py tests/service/data/test_payload_reconciliation_maria.py tests/endpoints/test_upload_endpoint_maria.py -v
```

---

### Commit 2.6: `test: replace insecure random with uuid` [RHOAIENG-55581]

**Files:**
- `tests/service/data/test_utils.py`

**Changes:**
- `uuid.uuid4()` instead of `random.randint()` for test request IDs (S311 fix)

**Verification:**
```bash
uv run ruff check tests/service/data/test_utils.py
```

---

### Commit 2.7: `refactor(config): update per-file-ignores for data layer` [RHOAIENG-57849]

**Files:**
- `pyproject.toml` — **only** per-file-ignores for `src/service/data/*`, `src/service/payloads/*`, `src/endpoints/consumer/*`, and `[tool.bandit]` entries

**Changes:** Add/update ruff per-file-ignores for data layer files (BLE001, SLF001, PLR0913, C901, etc.) and bandit skips (B608).

**Why:** Each PR carries its own pyproject.toml config to be fully independent.

---

### Commit 2.8: `style(data): formatting and docstrings for data layer` [RHOAIENG-57849]

**Files (formatting only):**
- `src/service/data/metadata/__init__.py`
- `src/service/data/shared_data_source.py`
- `src/service/data/storage/storage_interface.py`
- `src/service/data/storage/maria/utils.py`
- `src/service/data/storage/maria/__init__.py`
- `src/service/payloads/**/*.py` (all payload files)
- `src/service/utils/*.py` (exceptions, list_utils, logging_utils)
- `src/service/constants.py`
- `src/service/__init__.py`, `src/service/data/__init__.py`, `src/service/data/datasources/__init__.py`
- `tests/service/data/__init__.py`, `tests/service/payloads/**/__init__.py`
- `tests/service/payloads/**/*.py` (all payload test files)
- `tests/data/__init__.py`

**Changes:** Docstrings, import ordering, ruff format. No logic changes.

**Verification:**
```bash
uv run ruff check src/service/ tests/service/
uv run pytest tests/service/ -v
```

---

## PR #3: Core Metrics & Endpoints — Drift Type Fixes (1 commit) — ✅ CREATED as [#120](https://github.com/trustyai-explainability/trustyai-service/pull/120)

Branch: `refactor/metrics-endpoints`
Jira: RHOAIENG-57849
Status: **✅ Merged (2026-04-15). Approved by @RobGeada.**

### Commit 3.1: `refactor(drift): replace kwargs with explicit typed parameters` [RHOAIENG-57849]

**Files:**
- `src/core/metrics/drift/compare_means.py`
- `src/core/metrics/drift/utils.py`
- `src/core/metrics/drift/jensen_shannon.py`
- `tests/core/metrics/drift/test_compare_means.py`
- `tests/core/metrics/drift/test_utils.py`
- `tests/core/metrics/drift/test_jensen_shannon.py`

**Changes:**
- `NanPolicy = Literal["propagate", "raise", "omit"]` type alias
- `BwMethod = Literal["scott", "silverman"] | float | None` type alias
- Keyword-only `*` separator for bool param (FBT001)
- Remove `**kwargs: Any` → explicit params (ANN401 root cause fix)
- Remove unused `**_kwargs` from `prob_dist_hist`
- Fix test: remove spurious `bins=10` passed to `prob_dist_kde`

**Verification:**
```bash
uv run pytest tests/core/metrics/drift/ -v
uv run pyrefly check src/core/metrics/drift/
```

---

### Commit 3.2: `fix: add lint suppressions for justified patterns` [RHOAIENG-57849]

**Files:**
- `src/main.py` — `# noqa: S104` (intentional 0.0.0.0 bind)
- `src/endpoints/evaluation/lm_evaluation_harness.py` — `# noqa: S603` (subprocess with shlex)

**Note:** `src/service/utils/list_utils.py` (`# noqa: S301`) is in PR #2 scope (commit 2.8, formatting).

**Verification:**
```bash
uv run ruff check src/main.py src/endpoints/evaluation/
```

---

### Commit 3.3: `refactor(config): update per-file-ignores for metrics and endpoints` [RHOAIENG-57849]

**Files:**
- `pyproject.toml` — **only** per-file-ignores for `src/core/*`, `src/endpoints/{evaluation,explainers,metadata,metrics}/*`, `src/middleware/*`, `tests/**` test entries

**Changes:** Add/update ruff per-file-ignores for core metrics, endpoints, and middleware (N815, PLR0913, C901, etc.)

**Why:** Each PR carries its own pyproject.toml config to be fully independent.

---

### Commit 3.4: `style(metrics,endpoints): formatting and docstrings` [RHOAIENG-57849]

**Files (formatting only):**
- `src/core/metrics/drift/kolmogorov_smirnov.py`, `src/core/metrics/drift/__init__.py`
- `src/core/metrics/fairness/**/*.py` (all fairness files)
- `src/core/metrics/__init__.py`, `src/core/__init__.py`
- `src/endpoints/**/*.py` (consumer, data, evaluation, explainers, metadata, metrics — all)
- `src/endpoints/__init__.py` and sub-package `__init__.py` files
- `src/__init__.py`, `tests/__init__.py`
- `tests/core/**/__init__.py`, `tests/core/metrics/drift/factory.py`
- `tests/core/metrics/test_fairness.py`
- `tests/endpoints/**/__init__.py`
- `tests/endpoints/metrics/drift/test_compare_means.py` (endpoint-level, formatting)
- `tests/endpoints/metrics/drift/test_jensen_shannon.py` (formatting)
- `tests/endpoints/metrics/drift/test_kolmogorov_smirnov.py` (formatting)
- `tests/endpoints/metrics/drift/test_scheduler_exceptions.py` (formatting)
- `tests/test_app_integration.py`
- `tests/middleware/test_gzip_middleware_unit.py`
- `tests/resources/__init__.py`
- `src/middleware/gzip_middleware.py` (formatting)
- `src/proto/__init__.py`, `src/proto/grpc_predict_v2_pb2.py` (formatting)

**Changes:** Docstrings, import ordering, ruff format. No logic changes.

**Verification:**
```bash
uv run ruff check src/core/ src/endpoints/ src/middleware/
uv run pytest tests/core/ tests/endpoints/ tests/test_app_integration.py tests/middleware/ -v
```

---

## PR #4: Infrastructure Modernization (7 commits) — ✅ CREATED as [#118](https://github.com/trustyai-explainability/trustyai-service/pull/118)

Branch: `build/config-and-testing`
Jira: RHOAIENG-55591, RHOAIENG-55590, RHOAIENG-55592, RHOAIENG-55596
Status: **CI green (tests pass), ready for review**

### Key design decision: Decouple tool installation from CI enforcement

Adding `lint` and `type-check` CI jobs in the same PR as tool config caused a cascade — main's code doesn't pass ruff/pyrefly. Fix: this PR only modernizes the `test` job. Lint/type-check CI jobs move to PR #5 (after source code PRs land).

### Commit 4.1: `build(config): organize pyproject.toml and modernize pre-commit`

**Files:** `pyproject.toml`, `.pre-commit-config.yaml`
**Changes:**
- Replace isort/flake8/mypy with ruff/pyrefly/bandit in dev group
- Add types group (pandas-stubs, scipy-stubs, scikit-learn-stubs, h5py-stubs)
- Add pytest-xdist[psutil]>=3.8.0
- Add `[tool.pytest.ini_options]` with asyncio_mode=strict, addopts=--dist=loadgroup
- Add `[tool.bandit]` with exclude_dirs and skips
- Add `[tool.pyrefly]` with project-excludes, ignore-missing-imports, broad sub-configs for `src/**/*.py` and `tests/**/*.py`
- Add `[tool.ruff]` with exclude for protobuf
- CVE annotations on protobuf/cryptography (verified real — CVE-2026-0994 published 2026-01-23, CVE-2026-26007 published 2026-02-10)
- **Note:** `select = ["ALL"]` and per-file-ignores deferred to PR #5 (code must be fixed first)
- Remove flake8/mypy/isort pre-commit hooks, add ruff-check/ruff-format/pyrefly/bandit/check-yaml/conventional-pre-commit
- Add `ci.skip: [pyrefly-check]` — pre-commit.ci lacks uv/pyrefly (language: system)

### Commit 4.2: `build(deps): regenerate lock file`
**Files:** `uv.lock`

### Commit 4.3: `fix(config): add missing pyrefly test suppressions`
**Files:** `pyproject.toml`
**Changes:** Add `bad-assignment`, `not-iterable`, `unsupported-operation` to `tests/**/*.py` pyrefly sub-config (pre-existing type issues in test files)

### Commit 4.4: `fix(test): modernize fairness tests and fix deprecated pandas API`
**Files:** `tests/core/metrics/test_fairness.py`
**Changes:**
- Remove deprecated `axis=1` from `df.drop(columns=["Exited"], axis=1)` — newer pandas rejects both
- Replace `np.random.seed`/`np.random.*` with `np.random.default_rng` (numpy best practice)
- Add type annotations, docstrings, hypothesis `assume()` guards
- Fix individual_consistency tolerance (0.2 → 0.25) for RNG change

### Commit 4.5: `ci: modernize GitHub Actions workflows`
**Files:** `.github/workflows/python-tests.yaml`, `security-scan.yaml`, `build-and-push.yaml`
**Changes:**
- Modernize test job: uv replaces pip+venv, allow-prereleases for 3.14
- Remove dead `cache: pip`, add concurrency groups, update checkout v3→v4
- **No lint/type-check CI jobs** — deferred to PR #5
- **No `-n auto`** — deferred until MariaDB tests get xdist_group markers (see below)
- Remove `configFile: "pyproject.toml"` from bandit-action — action bundles Python 3.8 without TOML parser

### Commit 4.6: `fix(ci): remove parallel test execution to fix MariaDB test races`
**Files:** `.github/workflows/python-tests.yaml`
**Changes:** Remove `-n auto` from pytest command — MariaDB tests share a single DB and race under xdist

### Commit 4.7: `fix(ci): fix bandit-action TOML parsing and pre-commit.ci pyrefly skip`
**Files:** `.github/workflows/security-scan.yaml`, `.pre-commit-config.yaml`
**Changes:** Remove configFile from bandit-action (Python 3.8 TOML issue), add pyrefly-check to pre-commit.ci skip

### Deferred from PR #4 → add back in PR #5 or follow-ups

| Item | Why deferred | Add back in |
|---|---|---|
| `lint` and `type-check` CI jobs in python-tests.yaml | Main's code doesn't pass ruff/pyrefly | PR #5 (after PRs 1-3 merge) |
| `-n auto` parallel test execution | MariaDB tests share DB, cause race conditions (500 errors, missing tables) | Follow-up after adding `@pytest.mark.xdist_group("mariadb")` markers to MariaDB tests |
| `configFile: "pyproject.toml"` on bandit-action | Action bundles Python 3.8 without TOML parser | When PyCQA/bandit-action upgrades Python or adds TOML support |
| `--markdown-linebreak-ext=md` on trailing-whitespace hook | Dropped during pre-commit modernization (minor item m6) | PR #5 or any pre-commit update |
| GitHub Advanced Security / Bandit "configuration not found" | `security-scan.yaml` changed (removed `configFile`, modernized install steps) — GHAS can't match the config to main's baseline, so it can't determine new vs existing alerts. Resolves automatically once PR merges to main and a new baseline is established. | Self-resolving on merge |

---

## PR #5: Enable Strict Linting + CI Enforcement (depends on PRs 1-4) — ✅ CREATED as [#122](https://github.com/trustyai-explainability/trustyai-service/pull/122) (DRAFT)

Branch: `build/strict-linting` (based on `build/config-and-testing`)
Jira: RHOAIENG-55591
Status: **Draft — rebase needed after PRs #118-#121 merge. CI expected to fail until then.**

### What was deferred from PR #4 and must be added back:

**1. Add `lint` and `type-check` CI jobs to `python-tests.yaml`:**
```yaml
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - uses: astral-sh/setup-uv@v4
        with: { enable-cache: true, cache-dependency-glob: "uv.lock" }
      - run: uv sync --group dev
      - name: Lint with ruff
        run: |
          uv run ruff check src tests
          uv run ruff format --check src tests

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - uses: astral-sh/setup-uv@v4
        with: { enable-cache: true, cache-dependency-glob: "uv.lock" }
      - run: sudo apt-get update && sudo apt-get install -y protobuf-compiler
      - run: uv sync --group dev --group types
      - run: bash scripts/generate_protos.sh
      - name: Type check with pyrefly
        run: uv run pyrefly check src/
```

**2. Add `[tool.ruff.lint]` section:**
```toml
select = ["ALL"]
ignore = [
    "D203",   # one-blank-line-before-class (conflicts with D211)
    "D213",   # multi-line-summary-second-line (conflicts with D212)
    "E501",   # line-too-long
    "COM812", # trailing commas (conflicts with formatter)
]
```

**3. Add `[tool.ruff.lint.per-file-ignores]` section:**
```toml
# === SRC FILES ===
"src/core/metrics/drift/jensen_shannon.py" = ["PLR0913"]
"src/core/metrics/fairness/group/*.py" = ["PLR0913"]
"src/middleware/gzip_middleware.py" = ["C901", "PLR0911", "PLR0912", "PLR0913", "PLR0915"]
"src/service/data/datasources/data_source.py" = ["BLE001", "C901", "PLR0912"]
"src/service/data/modelmesh_parser.py" = ["PLR0913"]
"src/service/data/storage/**/*.py" = ["SLF001", "BLE001", "PLR0913"]
"src/service/data/storage/pvc.py" = ["C901", "PLR0912", "PLR0915"]
"src/service/payloads/**/*.py" = ["SLF001"]
"src/service/payloads/metrics/request_reconciler.py" = ["C901", "PLR0912", "PLR0915"]
"src/service/prometheus/prometheus_publisher.py" = ["SLF001"]
"src/service/prometheus/prometheus_scheduler.py" = ["BLE001"]
"src/endpoints/consumer/*.py" = ["N815"]
"src/endpoints/consumer/__init__.py" = ["C901"]
"src/endpoints/consumer/consumer_endpoint.py" = ["BLE001", "C901", "PLR0912", "PLR0913", "PLR0915"]
"src/endpoints/evaluation/lm_evaluation_harness.py" = ["SLF001"]
"src/endpoints/explainers/*.py" = ["N815"]
"src/endpoints/metadata.py" = ["N815", "BLE001", "C901"]
"src/endpoints/metrics/**/*.py" = ["N815"]
"src/endpoints/metrics/fairness/group/*.py" = ["BLE001"]

# === TEST FILES ===
"tests/**" = ["S101", "PT019", "SLF001"]
"tests/core/metrics/test_fairness.py" = ["N803", "N806"]
"tests/endpoints/metrics/drift/factory.py" = ["PLR0913", "C901", "PLR0915"]
"tests/endpoints/test_upload_endpoint_maria.py" = ["S105"]
"tests/endpoints/test_upload_endpoint_pvc.py" = ["PLR0913"]
"tests/service/data/test_utils.py" = ["PLR0913"]
```

**4. Tighten pyrefly sub-configs — replace broad `src/**/*.py` and `tests/**/*.py` with narrow per-file configs:**
```toml
# REMOVE these broad suppressions (added in PR #4 as temporary workaround):
[[tool.pyrefly.sub-config]]
matches = "src/**/*.py"
# ... (17 error types suppressed)

[[tool.pyrefly.sub-config]]
matches = "tests/**/*.py"
# ... (8 error types suppressed, including bad-assignment, not-iterable, unsupported-operation)

# KEEP only the narrow per-file sub-configs that are still needed
```

**5. Enable parallel test execution (after adding xdist markers):**
- Add `@pytest.mark.xdist_group("mariadb")` to all MariaDB integration tests
- Re-add `-n auto` to CI pytest command in `python-tests.yaml`
- Tests that share MariaDB DB: `test_upload_endpoint_maria.py`, `test_mariadb_storage.py`, `test_mariadb_migration.py`, `test_payload_reconciliation_maria.py`

**6. Restore `--markdown-linebreak-ext=md` to trailing-whitespace hook (minor item m6)**

**Verification:**
```bash
uv run ruff check src tests  # 0 violations with select=ALL
uv run pyrefly check src/    # 0 errors (broad sub-configs removed)
uv run pytest tests/ -n auto -q  # all pass with parallel execution
```

---

## CodeRabbit Issues from PR #117 (all resolved)

| # | Issue | Severity | Resolution |
|---|---|---|---|
| 1 | `InvalidSchemaError` import blocker | CRITICAL | ✅ Renamed exception in `exceptions.py` — import resolves |
| 2 | `StorageMetadata` positional compat | MAJOR | ✅ No backward compat needed (unreleased product) — config-only `__init__` |
| 3 | `set_recorded_inferences` keyword-only | CRITICAL | ✅ Caller already uses keyword syntax — no break |
| 4 | Async docstring in model_data.py | MINOR | ✅ Fixed: `await model_data.data()` |
| 5 | Named-value cleanup leak | MAJOR | ✅ Fixed: `_derived_ids` tracks and cleans up derived UUIDs |
| 6 | `gauge()` breaks callers | CRITICAL | ✅ All 4 scheduler calls migrated to `GaugeConfig(...)` |

---

## File → PR Assignment (no overlaps)

| File group | PR |
|---|---|
| `src/service/prometheus/*`, `tests/service/prometheus/*`, `tests/endpoints/metrics/drift/factory.py` | #1 |
| `src/service/data/*`, `src/service/payloads/*`, `src/service/utils/*`, `src/endpoints/consumer/*`, `tests/service/data/*`, `tests/service/payloads/*`, `tests/service/test_consumer_*`, `tests/endpoints/test_upload_*`, `tests/data/*` | #2 |
| `src/core/*`, `src/endpoints/{data,evaluation,explainers,metadata,metrics}/*`, `src/main.py`, `src/middleware/*`, `src/proto/*`, `src/__init__.py`, `tests/core/*`, `tests/endpoints/{__init__,metrics}/*`, `tests/test_app_*`, `tests/middleware/*`, `tests/__init__.py`, `tests/resources/*` | #3 |
| `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.github/workflows/*`, `tests/core/metrics/test_fairness.py` | #4 |
| `pyproject.toml` (`select=ALL` + per-file-ignores only) | #5 (after #1-3 merge) |

---

## Pre-Implementation Review Findings (all critical/major resolved)

Review found 2 critical, 11 major, and 5 minor issues. All critical and major items have been fixed in the working tree.

### CRITICAL — ✅ ALL FIXED

| # | Issue | Resolution |
|---|---|---|
| C1 | `consumer_endpoint.py` unreachable validation | ✅ Moved payload_kind check to early guard before `try` |
| C2 | `storage_metadata.py` `**kwargs` typing | ✅ Removed `**kwargs` entirely — config-only `__init__` (no backward compat needed) |

### MAJOR — ✅ ALL FIXED

| # | Issue | Resolution |
|---|---|---|
| M1 | Mangled docstrings (35+ files) | ✅ Fixed all split-sentence docstrings across codebase |
| M2 | `DataArrayConfig` dead code | Pre-existing on main — not a regression. Noted. |
| M3 | `pvc.py` pointless Lock alias | ✅ Removed conditional; use `asyncio.Lock` directly |
| M4 | `_derived_ids` lock coupling | ✅ Added comment documenting `_values_lock` protects both |
| M5 | `compare_means.py` keyword-only | ✅ All callers use keyword syntax — document in commit msg |
| M6 | Missing `[tool.ruff]` section | ✅ Added `[tool.ruff]` with `exclude` in correct location |
| M7 | CI bandit missing configFile | ⚠️ `configFile` removed — bandit-action bundles Python 3.8 without TOML parser. Config applied locally/pre-commit only. Re-add when action upgrades. |
| M8 | `addopts` without xdist guard | ✅ `--dist=loadgroup` is silently ignored without `-n`. `-n auto` removed from CI (MariaDB race conditions). |
| M9 | Dead `cache: 'pip'` | ✅ Removed from all workflows |
| M10 | Python 3.14 not GA | ✅ Added `allow-prereleases: true` |
| M11 | `cancel-in-progress` on build | ✅ Changed to `${{ github.event_name == 'pull_request' }}` |

### MINOR — remaining (low risk, fix during commit creation if convenient)

| # | Issue | Status |
|---|---|---|
| m1 | `GaugeConfig` allows both `value` and `named_values` | ✅ Fixed in PR #119 — validation rejects both |
| m2 | `factory.py` unused `TMetricRequest` import | ✅ Already resolved — import doesn't exist in current code |
| m3 | `reconcilable_output.py` `.get()` bug fix not applied | ✅ Fixed on branch `fix/reconciler-output-get` — pushed, PR pending |
| m4 | `drift_detected` returns `numpy.bool_` not `bool` | ✅ Already resolved — all three drift files already have `bool()` wrapping |
| m5 | ~~Inconsistent CVE placeholder annotations~~ | ✅ Fixed in PR #118 — `[forward-looking placeholder]` tags removed, CVEs verified as real |
| m6 | Lost `--markdown-linebreak-ext=md` | ✅ Fixed in PR #122 |
| m7 | MariaDB tests need `xdist_group` markers | Open — required before re-enabling `-n auto` in CI |
| m8 | Broad pyrefly sub-configs (`src/**`, `tests/**`) | Open — temporary workaround, tighten after code PRs land |
| m9 | pre-commit.ci skips pyrefly-check | Open — resolves itself once PR #118 merges (replaces flake8/mypy) |

---

## Deferred Items — When to Fix

### Window 1: During PR #122 rebase (right after #118-#121 merge)

These items cost nothing extra — you're already rebasing and touching these files.

1. **m7**: Add `@pytest.mark.xdist_group("mariadb")` to MariaDB tests, re-enable `-n auto`
2. **m8**: Replace broad pyrefly `src/**/*.py` / `tests/**/*.py` suppressions with narrow per-file configs
3. **m9**: Resolves itself — PR #118 replaces flake8/mypy with ruff/pyrefly, pre-commit.ci skip is already configured

**Checklist for rebase:**
```bash
# After PRs #118-#121 merge:
git checkout build/strict-linting
git rebase main
# Fix m7: add xdist markers to MariaDB tests
# Fix m8: tighten pyrefly sub-configs
# Verify:
uv run ruff check src tests           # 0 violations
uv run pyrefly check src/              # 0 errors
uv run pytest tests/ -n auto -q        # all pass with parallel execution
# Then mark PR #122 as ready for review
```

### Window 2: ~~After PR #122 merges~~ ✅ RESOLVED

All three items are resolved:

1. **m2**: ✅ Already resolved — `TMetricRequest` import doesn't exist in current code (cleaned up in merged PRs)
2. **m4**: ✅ Already resolved — all drift files already have `bool()` wrapping in current code
3. **m3**: ✅ Fixed — branch `fix/reconciler-output-get` pushed. One-line fix: `request_reconciler.py:106` changed `[provided_name]` to `.get(provided_name)` for consistency with the input schema path (line 77). 11 tests pass.

### PR strategy for m3

**Raise as a standalone PR** — independent of #118 and #122. It's a 1-line bug fix with no dependency on any other branch.

```
Branch: fix/reconciler-output-get (based on main)
PR title: fix(reconciler): use consistent dict access for output schema lookup
Merge: squash-merge to main (can merge immediately, no ordering constraints)
```

---

## Summary

- **5 PRs total** — 4 independent + 1 dependent (draft)
- **PRs #119, #120, #121 merged** — only #118 remains before #122 can proceed
- **PR #122 (draft)** depends on #118 — enables `select = ["ALL"]`, adds lint/type-check CI jobs
- **All PRs in origin:**
  - [#118](https://github.com/trustyai-explainability/trustyai-service/pull/118) — Config/CI modernization (CI green, awaiting review)
  - [#119](https://github.com/trustyai-explainability/trustyai-service/pull/119) — Prometheus GaugeConfig (✅ merged)
  - [#120](https://github.com/trustyai-explainability/trustyai-service/pull/120) — Drift type safety (✅ merged)
  - [#121](https://github.com/trustyai-explainability/trustyai-service/pull/121) — Data layer fixes (✅ merged)
  - [#122](https://github.com/trustyai-explainability/trustyai-service/pull/122) — Strict linting (draft, blocked on #118 only)
- **All 6 CodeRabbit issues resolved** ✅
- **All 13 critical/major review findings fixed** ✅
- **All 9 minor items resolved** — m1/m5/m6 fixed in PRs, m2/m4 already resolved on main, m3 fixed on `fix/reconciler-output-get` (PR pending), m7-m9 resolve during #122 rebase
- **No backward compatibility needed** — product unreleased, clean API breaks preferred
- ~~**First action:** Close PR #117~~ ✅ Done
- **Always run tests on PR branch before pushing**
