# New contributor guide

## pre-commit

In order to ensure proper and consistent code formatting, we use `pre-commit` in combination with tools like `ruff`, `flake8`, and `mypy`. When committing your code, make sure that all the pre-commit checks pass in your local environment.
To install pre-commit:

```bash
pre-commit install -t pre-commit -t commit-msg
```

Run manually pre-commit:

```bash
pre-commit run --all-files
```
