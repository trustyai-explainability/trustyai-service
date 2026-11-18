# OpenAPI document

[`openapi.yaml`](openapi.yaml) is the committed API contract. It is
`app.openapi()` of the FastAPI application, generated with every endpoint
feature flag pinned (explainers disabled) so it does not depend on deployment
configuration. CI fails if it is stale.

```bash
make generate-openapi  # regenerate
make check-openapi     # verify (CI runs the same check)
```

Use Python 3.12 to 3.14 and the `dev` dependency group. Regenerate and commit
the file whenever a route, request model or response model changes.
