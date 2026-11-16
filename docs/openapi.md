# OpenAPI document

[`openapi.yaml`](openapi.yaml) is the committed API contract. It is
`app.openapi()` of the FastAPI application, generated with every endpoint
feature flag pinned (explainers disabled) so it does not depend on deployment
configuration. CI fails if it is stale.

```bash
uv run python scripts/generate_openapi.py          # regenerate
uv run python scripts/generate_openapi.py --check  # verify (used in CI)
```

Use Python 3.12 to 3.14 and the `dev` dependency group. Regenerate and commit
the file whenever a route, request model or response model changes.
