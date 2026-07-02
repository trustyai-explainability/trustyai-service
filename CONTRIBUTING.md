# Contributing to TrustyAI Service

Thank you for your interest in contributing! This guide covers
the development setup, coding standards, and pull request process.

---

## Development Setup

### Prerequisites

- Python 3.12–3.14
- [uv](https://docs.astral.sh/uv/) (package manager)
- [pre-commit](https://pre-commit.com/)

### Install dependencies

```bash
uv sync --all-groups --extra mariadb
```

### Install pre-commit hooks

```bash
pre-commit install -t pre-commit -t commit-msg
```

This installs hooks for both code quality (pre-commit) and
commit message validation (commit-msg).

### Run the service locally

```bash
uv run python -m src.main
```

The API is available at `http://localhost:8080` and interactive
documentation at `http://localhost:8080/docs`.

---

## Code Style

- **Linting and formatting:**
  [Ruff](https://docs.astral.sh/ruff/) with `select = ["ALL"]`
  and targeted per-file ignores
- **Type checking:** [Pyrefly](https://pyrefly.org/)
- **Security scanning:**
  [Bandit](https://bandit.readthedocs.io/)
- **Secret detection:** detect-secrets, gitleaks
- **Line length:** 88 (Black default)
- **Type hints:** PEP 585/604 style — `dict[str, ...]` not
  `Dict`, `str | None` not `Optional[str]`
- **Pydantic:** v2

All checks run automatically on commit via pre-commit hooks.
To run them manually:

```bash
pre-commit run --all-files
```

---

## Testing

Run all tests:

```bash
uv run pytest tests/ -v
```

Skip tests that require a running MariaDB instance:

```bash
uv run pytest tests/ -v -k "not maria"
```

Generate a coverage report:

```bash
uv run pytest tests/ -v --cov=src --cov-report=xml
```

### Test conventions

- Mirror the `src/` directory structure under `tests/`
- Use `unittest.mock.patch` for mocking; `AsyncMock` for async
  methods
- New metrics should follow the test factory pattern used in
  [tests/endpoints/metrics/drift/factory.py](tests/endpoints/metrics/drift/factory.py)
  and
  [tests/core/metrics/drift/factory.py](tests/core/metrics/drift/factory.py)

---

## Commit Messages

We enforce
[Conventional Commits](https://conventionalcommits.org) via a
pre-commit hook. Format:

```text
type(scope): short description

Optional body with more detail.

Signed-off-by: Your Name <your.email@example.com>
```

### Types

| Type | Use for |
| ---- | ------- |
| `feat` | New feature or endpoint |
| `fix` | Bug fix |
| `refactor` | Code restructuring (no behavior change) |
| `test` | Adding or improving tests |
| `docs` | Documentation only |
| `chore` | Tooling, CI, dependencies |

### Sign-off requirement

All commits must include a `Signed-off-by` line (DCO). Use the
`--signoff` (or `-s`) flag:

```bash
git commit --signoff -m "feat(drift): add new metric"
```

---

## Pull Requests

### Branch naming

Use `type/short-description`:

```text
feat/tag-management-endpoint
fix/tls-crypto-policy
test/fairness-endpoint-tests
refactor/drop-modelmesh
docs/update-readme
chore/openssf-scorecard
```

### PR structure

Every PR should include:

- **Summary** — what changed and why
- **Test plan** — checklist of what was tested

Example:

```markdown
## Summary

- Add GET/POST /info/tags endpoints for tag management
- Fix tag constant mismatch bug in consumer_endpoint.py

## Test plan

- [x] 17 tests covering happy paths and edge cases
- [ ] CI passes
```

### Before submitting

1. All pre-commit hooks pass
2. Tests pass: `uv run pytest tests/ -v`
3. No unrelated changes (especially `uv.lock` drift)
4. Single commit preferred (squash if multiple)
5. Branch is based on `main` (rebase if needed)
6. PR targets the `main` branch

---

## Architecture

The service follows a three-layer structure:

```text
src/endpoints/    → FastAPI routers (HTTP handling)
src/core/         → Pure metric algorithms (no HTTP or storage)
src/service/      → Shared infrastructure (storage, scheduling)
```

**Boundary rule:** `src/core/` must not import from
`src/service/` or `src/endpoints/`. Core algorithms are
designed for eventual migration to the standalone
[trustyai-explainability-python](https://github.com/trustyai-explainability/trustyai-explainability-python)
library.

Each metric implements the standard 5-endpoint pattern:

1. Compute — `POST /metrics/...`
2. Definition — `GET /metrics/.../definition`
3. Schedule — `POST /metrics/.../request`
4. Delete — `DELETE /metrics/.../request`
5. List — `GET /metrics/.../requests`

All endpoint paths are defined as constants in
[src/endpoints/paths.py](src/endpoints/paths.py). New metrics
should add their paths there using the `MetricPaths` helper
rather than hardcoding strings in decorators.

---

## Security

Please report security vulnerabilities via
[GitHub Security Advisories](https://github.com/trustyai-explainability/trustyai-service/security/advisories/new),
not public issues. See [SECURITY.md](SECURITY.md) for details.

---

## License

By contributing, you agree that your contributions will be
licensed under the [Apache License 2.0](LICENSE).
