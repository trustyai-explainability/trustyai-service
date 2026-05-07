# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Safety Rules

**CRITICAL: NEVER run these commands under ANY circumstances:**

- `git reset --hard`
- `git reset` (without explicit user permission)
- `git checkout -- <file>`
- `git checkout .`
- `git restore <file>`
- `git restore .`
- `git clean -f`
- `git clean -fd`

**MANDATORY PROCEDURE before ANY git operation on modified files:**

When ANY of these occur:
- Thinking: "undo", "start fresh", "revert", "clean up"
- Considering: git restore/reset/checkout/clean/stash --drop
- Modified files exist (from previous git status)

**Execute these steps IN ORDER (no skipping):**

1. **STOP** - Pause immediately, recognize this is the danger pattern
2. **CHECK** - Run `git status` to see modified files
3. **REVIEW** - Run `git diff <file>` to see every line that would be lost
4. **SHOW USER** - Output the diff with description: "These changes would be permanently lost: [work description]"
5. **ASK** - "Discard [X lines] including [description]? (yes/no)" - Wait for explicit "yes"
6. **EXECUTE** - Only if user approves; otherwise STOP and ask what they want

**Rationale:** Uncommitted work represents days of effort. This procedure exists because prohibitions alone failed (4 incidents). See: `docs/ClaudeCode_git_safety_incident_report.md`

## Build & Run Commands

```bash
# Install all dependencies (dev + test + mariadb extra)
uv sync --all-groups --extra mariadb

# Run the service locally (dev mode)
uv run python -m src.main

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/core/metrics/drift/test_kolmogorov_smirnov.py -v

# Run a single test function
uv run pytest tests/core/metrics/drift/test_kolmogorov_smirnov.py::test_ks_no_drift -v

# Run tests excluding MariaDB (no DB setup needed)
uv run pytest tests/ -v -k "not maria"

# Run tests with coverage
uv run pytest tests/ -v --cov=src --cov-report=xml

# Lint (enforced in CI)
uv run ruff check src tests
uv run ruff format --check src tests

# Type checking (enforced in CI)
uv run pyrefly check

# Security scan
uv run bandit -c pyproject.toml -r src/ tests/

# Generate protobuf stubs (required after .proto changes)
bash scripts/generate_protos.sh

# Container build
podman build -t trustyai:latest --build-arg EXTRAS="mariadb,protobuf,eval" .
```

## Pre-commit Hooks

Pre-commit runs: merge conflict check, trailing whitespace, end-of-file fix, Python AST check, YAML/TOML check, ruff (check + format), pyrefly, bandit, detect-secrets, gitleaks, conventional commit message validation.

## Architecture

TrustyAI Service is a **REST API for AI model monitoring** (drift detection, fairness metrics, explainability) built on FastAPI + Hypercorn. It consumes inference data from KServe/ModelMesh, stores it, and computes metrics on schedule via Prometheus.

### Three-Layer Structure

```
src/endpoints/       → FastAPI routers (HTTP request/response handling)
src/core/            → Pure metric algorithms (no HTTP, no storage dependencies)
src/service/         → Shared infrastructure (storage, scheduling, data access)
src/serialization/   → JSON+gzip serialization (replaced pickle for CWE-502)
```

Each metric follows this pattern: a **core class** with static `calculate()` methods (in `src/core/metrics/`), an **endpoint router** that fetches data, calls the core, and returns results (in `src/endpoints/metrics/`), and optionally a **MetricsDirectory registration** so the Prometheus scheduler can discover it by name.

### Upstream Migration Path

**IMPORTANT:** `src/core/` is a staging area for code destined for [trustyai-explainability-python](https://github.com/trustyai-explainability/trustyai-explainability-python).

This code will be migrated upstream to create a standalone Python library of metric algorithms. Once migrated, this service will import from `trustyai-explainability` instead of maintaining its own implementations.

**Boundary enforcement:**
- `src/core/` **MUST NOT** import from `src/service/` or `src/endpoints/`
- `src/core/` **MUST NOT** depend on FastAPI, storage interfaces, or Prometheus
- Keep core logic pure: algorithms only, no infrastructure dependencies
- When reviewing changes to `src/core/`, verify they could run standalone in the upstream library

**Migration workflow:**
1. Implement algorithms in `src/core/metrics/` with minimal dependencies
2. Wrap with HTTP/storage in `src/endpoints/` and `src/service/`
3. Periodically upstream `src/core/` implementations to trustyai-explainability-python
4. Replace local implementations with upstream imports

Notable core data structures: `src/core/metrics/drift/greenwald_khanna_quantile_sketch.py` implements a streaming quantile sketch used by the streaming KS test (`/metrics/drift/ksteststreaming`) for memory-efficient drift detection.

### Data Flow

1. **Ingest**: KServe consumer (`POST /consumer/kserve/v2`) or data upload (`POST /data/upload`) receives inference payloads
2. **Decompress**: `GzipRequestMiddleware` handles gzip-encoded requests
3. **Reconcile**: Input/output payloads matched by request ID via `consume_cloud_event()`
4. **Store**: Serialized as JSON+gzip (via `src/service/serialization/`), persisted via storage interface (PVC/HDF5 or MariaDB)
5. **Compute**: Background `asyncio` task runs `PrometheusScheduler.calculate()` every 30s for scheduled metrics
6. **Expose**: REST endpoints return on-demand metric values; Prometheus scrapes `GET /q/metrics`

### Storage Backends

Controlled by `SERVICE_STORAGE_FORMAT` env var:
- **PVC** (default): HDF5 files on persistent volumes. Config: `STORAGE_DATA_FOLDER`, `STORAGE_DATA_FILENAME`
- **MariaDB** (optional, requires `mariadb` extra): Config via `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USERNAME`, `DATABASE_PASSWORD`, `DATABASE_DATABASE`

Backend selection is in `src/service/data/storage/__init__.py`. Both implement the async `StorageInterface` in `storage_interface.py`.

### Singletons

Three shared service instances accessed via getter functions:
- `get_storage_interface()` — storage backend
- `get_shared_data_source()` — data access layer wrapping storage
- `get_shared_prometheus_scheduler()` — recurring metric computation

### Endpoint Conventions

The **CompareMeans drift endpoint** (`src/endpoints/metrics/drift/compare_means.py`) is the canonical template for new metrics. It uses:
- Per-metric Pydantic request model inheriting `BaseMetricRequest`
- `@model_validator(mode="after")` to auto-set `metric_name`
- Default constants imported from the core module
- Return type annotations on all endpoints
- `METRIC_NAME` constant instead of string literals
- `deprecated=True` + `log_deprecated_endpoint()` for backward-compatible aliases
- Standard 5-endpoint pattern: compute (POST), definition (GET), schedule (POST .../request), delete (DELETE .../request), list (GET .../requests)

See `docs/fairness-vs-drift-endpoints.md` for a detailed comparison of endpoint patterns and the target convergence design.

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVICE_STORAGE_FORMAT` | `PVC` | Storage backend (`PVC` or `MARIA`) |
| `SERVICE_METRICS_SCHEDULE` | `30` | Seconds between scheduled metric computations |
| `HTTP_PORT` | `8080` | HTTP listener port |
| `SSL_PORT` | `4443` | HTTPS listener port |
| `TLS_CERT_FILE` | `/etc/tls/internal/tls.crt` | TLS certificate path |
| `TLS_KEY_FILE` | `/etc/tls/internal/tls.key` | TLS private key path |

### Health & Metrics Endpoints

- `GET /q/health/ready` — readiness probe
- `GET /q/health/live` — liveness probe
- `GET /q/metrics` — Prometheus metrics (OpenMetrics format)
- `GET /docs` — OpenAPI documentation (FastAPI auto-generated)

## Code Style

- **Linting**: ruff with `select = ["ALL"]` and targeted per-file-ignores (see `pyproject.toml`)
- **Line length**: 88 (black default)
- **Type checking**: pyrefly (replaces mypy)
- **Security**: bandit (configured in `pyproject.toml`)
- **Python version**: 3.12-3.14 (use `dict[str, ...]` not `Dict`, `str | None` not `Optional[str]`)
- **Pydantic**: V2 with `ConfigDict` for model configuration, `Field(alias=...)` for camelCase API compatibility
- **Commit messages**: Conventional Commits format enforced by pre-commit (`feat:`, `fix:`, `refactor:`, etc.)
