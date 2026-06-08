# PR #122 Critical Review: `build/strict-linting`

Systematic file-by-file diff against `origin/main`. Each finding categorized as:
- **COSMETIC**: Pure formatting/lint fix (line wrapping, docstring style, import sorting)
- **BEHAVIORAL**: Changes that alter runtime behavior
- **QUESTIONABLE**: Changes that may be unnecessary, incorrect, or introduce risk

---

## 1. CI/Config/Build Files

### `.github/workflows/ci-build.yaml`
- **BEHAVIORAL**: Changed `sed` pattern for Containerfile label injection. Old pattern matched `summary="odh-trustyai-service-python"`, new matches `io.trustyai.fips.documentation="..."`. Presumably the Containerfile changed and the old pattern no longer matched, but this needs verification against the actual Containerfile.

### `.github/workflows/python-tests.yaml`
- **BEHAVIORAL**: Added `permissions: contents: read` (least-privilege). Good security practice.
- **BEHAVIORAL**: Added separate `lint` and `type-check` jobs (ruff check/format, pyrefly). Previously only `test` job existed. This is a real improvement — catches lint/type issues earlier in CI.

### `.pre-commit-config.yaml`
- **COSMETIC**: Added `--markdown-linebreak-ext=md` to trailing-whitespace hook.
- **BEHAVIORAL**: Upgraded ruff from v0.15.10 → v0.15.12. Expected for a linting PR.
- **QUESTIONABLE**: **Downgraded gitleaks from v8.30.1 → v8.30.0**. Why downgrade a security tool? This should be v8.30.1 or later unless there's a specific compatibility issue.

### `.sourcery.yaml`
- **COSMETIC**: Deleted empty file. Fine.

### `CLAUDE.md`
- **COSMETIC**: New file. Project documentation for Claude Code. Not shipped to production.

### `pyproject.toml`
- **BEHAVIORAL**: Added `bandit-sarif-formatter` dev dependency. Reasonable.
- **BEHAVIORAL**: Added `[tool.ruff.lint]` section with `select = ["ALL"]` and per-file-ignores. This is the core purpose of the PR.
- **BEHAVIORAL**: Added `src.proto.grpc_predict_v2_pb2.*` to pyrefly `ignore-missing-imports`. Reasonable.
- **BEHAVIORAL**: Removed pyrefly sub-config for `tests/service/data/test_utils.py` that suppressed `not-a-type`. Need to verify this doesn't reintroduce type errors.

### `scripts/generate_protos.sh`
- **COSMETIC**: Added comments about pyrefly exclusion.

### `scripts/test_upload_endpoint.sh`
- **COSMETIC**: Added missing trailing newline.

### `uv.lock`
- Auto-generated. No review needed.

---

## 2. `src/core/` — Core Metric Algorithms

### `src/core/__init__.py`, `src/core/metrics/__init__.py`, etc. (7 files)
- **COSMETIC**: Added module docstrings to previously empty `__init__.py` files. Required by ruff D104.

### `src/core/metrics/drift/compare_means.py`
- **COSMETIC**: Docstring style (multi-line → single-line summary), line wrapping.
- **COSMETIC**: `NanPolicy` type alias moved below constants (still before its first use in `DEFAULT_NAN_POLICY`). OK.
- **BEHAVIORAL**: `alpha` parameter moved from positional to keyword-only (after `*`). All callers use `alpha=alpha`, so backward-compatible. But this is a public API change that could break external callers passing positionally.
- **BEHAVIORAL**: `raise ValueError("...")` → `msg = "..."; raise ValueError(msg)`. Required by ruff EM101. Mechanically correct.
- **BEHAVIORAL**: `statistic, p_value = stats.ttest_ind(...)` → `result = stats.ttest_ind(...); statistic = float(result.statistic); p_value = float(result.pvalue)`. Uses named tuple attributes instead of positional unpacking. Good — more explicit and adds `float()` conversion.
- **BEHAVIORAL**: Removed `bool()` wrapper from `drift_detected: bool(p_value < alpha)` → `p_value < alpha`. **Safe** because both operands are now Python `float` after the `float()` casts above.

### `src/core/metrics/drift/jensen_shannon.py`
- **COSMETIC**: Docstring style, line wrapping.
- **BEHAVIORAL**: Removed `from .utils import BwMethod` — now uses `utils.BwMethod` inline. Functionally identical.
- **BEHAVIORAL**: `bw_method` moved from keyword-only (after `*`) to positional. All callers use `bw_method=bw_method`. **Opposite direction** from compare_means — less strict, which is a loosening of the API contract.
- **BEHAVIORAL**: `distance = jensenshannon(...)` → `distance_raw = jensenshannon(...); distance = float(distance_raw); divergence = float(distance_raw**2)`. Adds explicit `float()` conversion. Good.
- **BEHAVIORAL**: Removed `bool()` wrapper from `drift_detected`. Safe (same reasoning as compare_means).
- **COSMETIC**: `raise ValueError("msg")` → `msg = "msg"; raise ValueError(msg)` (EM101).

### `src/core/metrics/drift/kolmogorov_smirnov.py`
- **COSMETIC**: Docstring style, `Dict` → `dict` (modern Python).
- **BEHAVIORAL**: Same pattern as compare_means: named tuple access + `float()` cast + `bool()` removal. All safe.
- **COSMETIC**: `raise ValueError("msg")` → `msg/raise` pattern.

### `src/core/metrics/drift/utils.py`
- **COSMETIC**: Docstring style, line wrapping.
- **BEHAVIORAL**: `BwMethod` type alias moved above `DEFAULT_BINS`/`DEFAULT_GRID_POINTS`. Just reordering, still before use.
- **BEHAVIORAL**: Added `MIN_KDE_SAMPLE_SIZE = 2` constant replacing magic number `2`. Good.
- **BEHAVIORAL**: `bw_method` in `prob_dist_kde` moved from keyword-only to positional. Same as jensen_shannon change.
- **BEHAVIORAL**: Return type changed from `return [p_x, p_y]` (list) to `return np.array([p_x, p_y])` (ndarray). The function's return type annotation says `-> np.ndarray`, so this **fixes a type annotation lie**. Callers unpack as `p_ref, p_cur = prob_dist_kde(...)` which works identically for both list and ndarray.
- **COSMETIC**: `raise ValueError("msg")` → `msg/raise` pattern throughout.

### `src/core/metrics/fairness/fairness_metrics_utils.py`
- **COSMETIC**: Added module docstring, function docstrings, `Callable` import moved from `typing` to `collections.abc`.
- **COSMETIC**: Line wrapping, `raise ValueError("msg")` → `msg/raise` pattern.
- No behavioral changes.

### `src/core/metrics/fairness/group/disparate_impact_ratio.py`
- **COSMETIC**: Added module docstring, line wrapping.
- **QUESTIONABLE**: **Docstrings are mangled**. Formatter joined separate `:param` entries into continuous text. Example:
  ```
  :param model the model to be tested for fairness : param
      privilege_columns a list of integers specifying the indices
  ```
  The `: param` mid-line is not a valid RST parameter directive. This breaks documentation tooling (Sphinx, pydoc).

### `src/core/metrics/fairness/group/group_average_odds_difference.py`
- **COSMETIC**: `List[int]` → `list[int]`, added type annotations to `calculate()`, `privilege_filter()`.
- **BEHAVIORAL**: `postive_class: List[int]` → `postive_class: int`. The parameter type was changed from list to int. Need to verify callers.
- **QUESTIONABLE**: **Same mangled docstrings** as DIR above. Multiple `:param` entries merged.
- **BEHAVIORAL**: Added `-> float` return type to `calculate()`. Was untyped before. This is a correctness fix.

### `src/core/metrics/fairness/group/group_average_predictive_value_difference.py`
- Same pattern as AOD above.
- **QUESTIONABLE**: **Same mangled docstrings**.

### `src/core/metrics/fairness/group/group_statistical_parity_difference.py`
- Same pattern as DIR above.
- **QUESTIONABLE**: **Same mangled docstrings**.

### `src/core/metrics/fairness/individual/individual_consistency.py`
- **BEHAVIORAL**: Complete rewrite of calculation logic. Changed from "subtract fractions per mismatch" to "count mismatches then divide". Fixes numerical stability issues and scalar/array compatibility.
- **BEHAVIORAL**: Added `proximity_function` type annotation from `Any` to `Callable[[np.ndarray, np.ndarray], np.ndarray]`.
- **BEHAVIORAL**: Added input validation (empty samples, empty model output, zero output_width, no neighbors).
- **BEHAVIORAL**: Changed comparison from `!=` to `np.array_equal()` for NumPy 2.0 compatibility.
- **BEHAVIORAL**: Uses `np.size()` instead of `len()` for scalar handling.
- Overall: This is a **genuine bug fix**, not just a lint fix. Well done but significant behavioral change bundled in a "linting" PR.

---

## 3. `src/main.py`

- **COSMETIC**: Added module docstring, function docstrings, line wrapping.
- **BEHAVIORAL**: LM eval harness import refactored from boolean flag pattern to `Optional[APIRouter]` pattern. Cleaner. The bare `import router` in the try block doesn't conflict because it's immediately assigned to `lm_evaluation_harness_router`.
- **BEHAVIORAL**: Removed commented-out imports (`# from fastapi_utils.tasks`, `# from src.endpoints.explainers`, `# from src.endpoints.drift_metrics`, `# app.include_router(explainers_router`). Good — dead code removal.
- **COSMETIC**: `app` → `_app` in lifespan function (unused parameter convention).
- **COSMETIC**: `request` → `_request` in metrics endpoint (unused parameter).
- **BEHAVIORAL**: Health probe status codes: `status_code=200` → `status_code=HTTPStatus.OK`. Functionally identical (200).
- **COSMETIC**: f-string logger calls → `%s` style (ruff G004). Correct.
- **BEHAVIORAL**: `host_https = "0.0.0.0"` gets `# noqa: S104` (intentional binding).
- **COSMETIC**: Removed `else` after `return` in `get_tls_config()` (RET505).
- **BEHAVIORAL**: `TYPE_CHECKING` guard for `APIRouter` import. Avoids runtime import for type-only usage.

---

## 4. `src/middleware/gzip_middleware.py`

- **COSMETIC**: Docstring style, line wrapping throughout.
- **BEHAVIORAL (OK)**: `_should_process_path` → `should_process_path` and `_should_process_content_type` → `should_process_content_type`. Removes underscore prefix (private → public). Reasonable — these are stable, simple predicates useful for callers/tests, and were already tested directly (making the `_` prefix a lie). `_decompress_body` correctly stays private.
- **BEHAVIORAL**: `fail_on_error` moved from positional to keyword-only. All callers use keyword syntax. Backward-compatible.
- **BEHAVIORAL**: `content_type.split(";")` → `content_type.split(";", maxsplit=1)`. Micro-optimization, functionally equivalent.
- **COSMETIC**: f-string logger → `%s` style throughout.
- **COSMETIC**: `raise ValueError("msg")` → `msg/raise` pattern.

---

## 5. `src/endpoints/consumer/__init__.py`

- **BEHAVIORAL**: `Optional[str]` → `str | None`, `Dict` → `dict`, `List` → `list`. Modern Python syntax.
- **BEHAVIORAL**: `str, Enum` → `StrEnum`. More Pythonic for string enums.
- **BEHAVIORAL**: `get_prediction_id() -> str` → `-> str | None`. **Correctness fix** — field is Optional, so return was lying.
- **BEHAVIORAL**: `get_kind() -> PartialKind` → `-> PartialKind | None`. Same fix.
- **BEHAVIORAL**: `id` parameter renamed to `id_` (avoids shadowing builtin). Good.
- **BEHAVIORAL**: Added `InferParameter` class. Was this moved from elsewhere? Need to check.
- **COSMETIC**: Added docstrings to all classes and methods.

---

## Summary of Issues Found

### MUST FIX (broken by the PR)

1. **Mangled docstrings in 4 fairness core files** — `:param` entries merged into unreadable blobs. Affects:
   - `src/core/metrics/fairness/group/disparate_impact_ratio.py`
   - `src/core/metrics/fairness/group/group_average_odds_difference.py`
   - `src/core/metrics/fairness/group/group_average_predictive_value_difference.py`
   - `src/core/metrics/fairness/group/group_statistical_parity_difference.py`

### SHOULD INVESTIGATE

2. **Gitleaks downgraded** v8.30.1 → v8.30.0 in `.pre-commit-config.yaml`. Why?
3. **`postive_class` type change** in AOD: `List[int]` → `int`. Verify this doesn't break callers.
4. **`bw_method` keyword-only → positional** in `jensen_shannon.py` and `utils.py` — loosens API contract (opposite direction from `compare_means.py` which tightened it).

### ACCEPTABLE BUT NOTABLE

6. **Individual consistency rewrite** is a genuine algorithmic fix bundled in a linting PR. Consider calling this out in the PR description.
7. **`return [p_x, p_y]` → `return np.array([p_x, p_y])`** in drift utils — fixes type annotation lie, backward-compatible.
8. **`bool()` removal** from drift metrics — safe because `float()` casts were added upstream.
9. **LM eval import refactor** in main.py — cleaner pattern.
10. **Return type corrections** in consumer `__init__.py` — fixes lies in type annotations.

---

## 6. `src/endpoints/` — REST API Endpoints

### `src/endpoints/evaluation/lm_evaluation_harness.py` (most significant changes)
- **BEHAVIORAL (BUG FIX)**: Environment variables (`env_vars`) were logged but **never actually passed** to `subprocess.Popen`. Now properly merged and passed via `env=merged_env`.
- **BEHAVIORAL (BUG FIX)**: Boolean CLI flags (`_StoreTrueAction`/`_StoreFalseAction`) were always emitted regardless of the field value. Now only emitted when value matches the action type.
- **BEHAVIORAL (BUG FIX)**: POST `/job` route was missing `API_PREFIX`, inconsistent with all other endpoints. Fixed.
- **BEHAVIORAL (BUG FIX)**: DELETE route had `{id}` instead of `{job_id}`, causing a FastAPI parameter binding bug. Fixed.
- **BEHAVIORAL**: Full thread-safety overhaul — `job_registry_lock`, background reader threads, atomic state transitions. Substantial correctness improvement.
- **BEHAVIORAL**: `delete_all_lm_eval_job()` now only removes the specific jobs stopped (was `job_registry.clear()` — race condition).
- **BEHAVIORAL**: `stop_lm_eval_job()` now waits for process termination with timeout + SIGKILL fallback.
- **BEHAVIORAL**: `JobIdGenerator` class replaces `global LAST_ID` pattern. Thread-safe.

### `src/endpoints/consumer/consumer_endpoint.py`
- **BEHAVIORAL (BUG FIX)**: `consume_cloud_event`: `payload.id = ce_id` no longer overwrites a valid payload ID with `None` when CE header is absent.
- **BEHAVIORAL (SECURITY)**: Error messages in 500 responses no longer leak `str(e)` to clients.
- **BEHAVIORAL**: Added `_validate_payload_type()` for defensive type checking.
- **BEHAVIORAL**: `except Exception` narrowed to `except ValueError` for payload parsing.
- **COSMETIC**: f-string logger → `%s` style, `datetime.now(timezone.utc)` → `datetime.now(UTC)`.

### `src/endpoints/consumer/__init__.py`
- **BEHAVIORAL**: `KServeData.parameters` type changed from `dict[str, str] | None` to `dict[str, InferParameter | str] | None`. New `InferParameter` class added. Could affect deserialization if existing payloads use string parameters.
- **BEHAVIORAL**: `ValueError` → `TypeError` for BYTES datatype validation. Safe within Pydantic validators.
- **BEHAVIORAL**: `model_name: str = None` → `str | None = None`. Correct fix.

### `src/endpoints/metrics/drift/compare_means.py`
- **BEHAVIORAL**: `nan_policy` field type changed from `str` to `NanPolicy` enum. Adds proper validation.
- **BEHAVIORAL**: `batch_size` gets `gt=0` constraint; `alpha` gets `gt=0, lt=1`.
- **BEHAVIORAL**: Scheduler unavailability returns 503 instead of 500.
- **BEHAVIORAL**: Input validation added before try block (reference_tag, fit_columns whitespace stripping).

### `src/endpoints/metrics/drift/jensen_shannon.py`, `kolmogorov_smirnov.py`
- Same patterns as compare_means. All correct.

### `src/endpoints/metrics/fairness/group/dir.py`, `spd.py`
- **BEHAVIORAL**: Exception handling restructured (TRY301 avoidance). Correct but control flow less obvious.
- **BEHAVIORAL**: Delta validation added: rejects negative values.
- **COSMETIC**: `Query(None)` → `Annotated[float | None, Query()]` style.

### `src/endpoints/metrics/fairness/group/utils.py`
- **BEHAVIORAL**: `GroupDefinitionRequest` rewritten from camelCase fields + `@property` to `Field(alias=...)`. API-compatible due to `populate_by_name=True`. Safe because `GroupDefinitionRequest` is only used in unimplemented endpoints (501).
- **BEHAVIORAL**: New `_extract_values()` helper replaces inline value extraction. More robust (handles `None`, validates scalar-compatibility).

### `src/endpoints/explainers/global_explainer.py`, `local_explainer.py`
- **BEHAVIORAL**: Unimplemented endpoints now raise 501 NOT_IMPLEMENTED instead of returning dummy data. Correct.

### `src/endpoints/metadata.py`
- **QUESTIONABLE**: `type` parameter renamed to `inference_type`. Changes query parameter name. Low risk (endpoint is unimplemented/501).

### `src/endpoints/metrics/metrics_info.py`
- **QUESTIONABLE**: `type` renamed to `type_`. Creates ugly `?type_=...` query parameter. Should use `Query(alias="type")` when implemented.
- **QUESTIONABLE**: Mangled module docstring: "retrieving scheduled metric." / "computations."

### `src/endpoints/data/data_upload.py`
- **QUESTIONABLE**: Mangled module docstring: "TrustyAI." / "service."
- **BEHAVIORAL (SECURITY)**: Error messages no longer leak `str(e)` to clients.

---

## 7. `src/service/` — Shared Infrastructure

### `src/service/data/datasources/data_source.py`
- **QUESTIONABLE (POTENTIAL BUG)**: `df[df[UNLABELED_TAG] != True]` → `df[~df[UNLABELED_TAG]]`. These are **NOT equivalent** when the column contains `None`/`NaN` values — `~` raises `TypeError` on `None`, while `!= True` treats `None` as "not True" (keeps the row). If the UNLABELED_TAG column is always boolean, this is fine. If it can contain `None`, this will crash at runtime.
- **BEHAVIORAL**: `get_metadata` callers at lines 258, 271 type-annotate the return as `StorageMetadata` but the method now returns `StorageMetadata | None`. If `None` is returned, the subsequent `.get_observations()` / `.is_recorded_inferences()` calls will crash with `AttributeError`.

### `src/service/data/storage/pvc.py`
- **BEHAVIORAL**: `get_metadata` return type changed from `StorageMetadata` to `StorageMetadata | None`. Returns `None` on error instead of empty `StorageMetadata`. Callers not fully updated (see data_source.py above).

### `src/service/data/storage/storage_interface.py`
- **BEHAVIORAL**: `get_metadata` ABC signature changed from `-> Dict` to `-> StorageMetadata | None`.

### `src/service/data/exceptions.py`
- **BEHAVIORAL**: Exception classes renamed: `DataframeCreateException` → `DataframeCreateError`, `StorageReadException` → `StorageReadError`. All references updated.

### `src/service/data/metadata/storage_metadata.py`
- **BEHAVIORAL**: `StorageMetadata` now takes a `StorageMetadataConfig` object in constructor instead of individual parameters. All callers updated.

### `src/service/prometheus/gauge_config.py`
- **QUESTIONABLE**: Removed validation that rejected providing both `value` AND `named_values`. Now allows both, but `PrometheusPublisher.gauge()` silently ignores `named_values` when `value` is present. Was an explicit error, now silent.

### `src/service/prometheus/prometheus_publisher.py`
- **QUESTIONABLE**: Old derived gauge values no longer cleaned up on recalculation cycles — only on removal. Could lead to unbounded growth of `self.values` dict if a metric with `named_values` is recalculated repeatedly.

### `src/service/prometheus/shared_prometheus_scheduler.py`
- **BEHAVIORAL**: Refactored from `global` variable to singleton class. Correct pattern. **No thread safety lock** unlike `shared_data_source.py`. Inconsistent, but likely fine since scheduler is created during startup.

### `src/service/data/storage/maria/legacy_maria_reader.py`
- **QUESTIONABLE**: Mangled module docstring: "Java-." / "serialized data."

### `src/service/data/shared_data_source.py`
- **BEHAVIORAL**: Refactored from `global` variable to singleton class with `threading.Lock`. Good thread safety improvement.

### `src/service/payloads/service/schema_item.py`
- **QUESTIONABLE**: `set_type` has overly complex dual-parameter approach (`data_type` positional, `type` keyword-only) for a single-caller method.

---

## 8. `tests/` — Test Files

### Test Bugs
- **BUG**: 3 test method names mangled in `tests/middleware/test_gzip_middleware_unit.py`: `testshould_process_path` (missing underscore after `test`), `testshould_process_content_type_exact_and_parameters`, `testshould_process_content_type_patterns`. Tests still run (pytest discovers them) but names are wrong.
- **BUG**: Mangled docstrings:
  - `tests/endpoints/metrics/drift/test_scheduler_exceptions.py`: "surfaced." / "by endpoints." (should be one line)
  - `tests/endpoints/metrics/drift/factory.py`: "valid and. malformed" (stray period)
  - `tests/endpoints/test_upload_endpoint_pvc.py`: "dimensions and. datatypes" (stray period)
- **BUG**: Copy-paste docstring error: `test_spd_range` says "DIR calculation is always positive" — should say "SPD calculation is within expected range".

### Notable Behavioral Changes
- **`np.random.seed()` → `np.random.default_rng()`**: Different random sequences but tests only check structural properties, not exact values. Safe.
- **8 gzip integration tests removed** from `test_upload_endpoint_pvc.py`. Unit test coverage exists in `test_gzip_middleware_unit.py` but integration-level coverage is lost.
- **Tolerance tightened**: `pytest.approx(1.0, abs=0.01)` → `abs=1e-5` in fairness test. Much stricter — verify it doesn't flake.
- **`self.assertEqual` → `assert`**: Throughout upload/MariaDB tests. Functionally correct with pytest rewriting.
- **Exception class renames**: `DataframeCreateException` → `DataframeCreateError`, etc. Matches source changes.
- **`StorageMetadata` constructor**: Now wraps args in `StorageMetadataConfig`. Matches source refactor.
- **HTTP status codes**: Various expected codes updated (500→400 for invalid UUID, 500→503 for scheduler unavailable). All verified correct.

### Cosmetic (correct, no risk)
- Module docstrings added to all test files and `__init__.py`
- `-> None` return type annotations on all test methods
- Line wrapping at 88 chars
- Magic number extraction to named constants (over-engineered in some cases: `HIST_BINS_10 = 10`)
- `os.path` → `pathlib.Path`
- `print()` debug statements removed from tearDown

---

## Consolidated Issue List

### MUST FIX

| # | Severity | File(s) | Issue |
|---|----------|---------|-------|
| 1 | HIGH | 4 fairness core files | Mangled `:param` docstrings — params merged into unreadable blobs |
| 2 | HIGH | `data_source.py:258,271` | `get_metadata()` callers don't handle new `None` return — will crash with `AttributeError` |
| 3 | MEDIUM | `data_source.py:172` | `~df[UNLABELED_TAG]` not equivalent to `df[col] != True` for `None`/`NaN` values |
| 4 | MEDIUM | `test_gzip_middleware_unit.py` | 3 test method names mangled (missing `_` after `test`) |
| 5 | LOW | 4 test/src files | Mangled docstrings with stray periods or broken line wrapping |
| 6 | LOW | `test_fairness.py` | Copy-paste error: `test_spd_range` docstring says "DIR" |

### SHOULD INVESTIGATE

| # | File(s) | Issue |
|---|---------|-------|
| 7 | `.pre-commit-config.yaml` | Gitleaks downgraded v8.30.1 → v8.30.0. Why? |
| 8 | `gauge_config.py` | Removed validation rejecting both `value` AND `named_values` |
| 9 | `prometheus_publisher.py` | Potential memory leak: derived gauge values not cleaned up on recalculation |
| 10 | `group_average_odds_difference.py` | `postive_class` type changed `List[int]` → `int`. Verify callers. |
| 11 | `test_upload_endpoint_pvc.py` | 8 gzip integration tests removed. Sufficient unit coverage? |
| 12 | `test_fairness.py` | Tolerance tightened from 0.01 to 1e-5. Verify no flakiness. |

### QUESTIONABLE BUT LOW RISK

| # | File(s) | Issue |
|---|---------|-------|
| 13 | `jensen_shannon.py`, `utils.py` | `bw_method` loosened from keyword-only to positional |
| 15 | `metrics_info.py` | `type` → `type_` creates ugly query parameter |
| 16 | `metadata.py` | `type` → `inference_type` changes query parameter name |
| 17 | `legacy_maria_reader.py` | Mangled module docstring |
| 18 | `schema_item.py` | Overly complex `set_type` dual-parameter approach |
| 19 | `shared_prometheus_scheduler.py` | No thread safety lock (inconsistent with `shared_data_source.py`) |

### GENUINE BUG FIXES (good, but notable for PR scope)

| # | File(s) | Fix |
|---|---------|-----|
| 20 | `lm_evaluation_harness.py` | env_vars actually passed to subprocess |
| 21 | `lm_evaluation_harness.py` | Boolean CLI flag logic fixed |
| 22 | `lm_evaluation_harness.py` | POST `/job` route path fixed |
| 23 | `lm_evaluation_harness.py` | DELETE `{id}` → `{job_id}` parameter binding fixed |
| 24 | `lm_evaluation_harness.py` | Full thread-safety overhaul |
| 25 | `consumer_endpoint.py` | `ce_id` null-propagation fix |
| 26 | `individual_consistency.py` | Algorithm rewrite (numerical stability + NumPy 2.0) |
| 27 | `consumer/__init__.py` | Return type annotations corrected (`str` → `str | None`) |
| 28 | Various | Security hardening: error messages no longer leak to clients |
