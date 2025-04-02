# New contributor guide

## Installation

TrustyAIService uses `uv` as a package manager. To install it, follow the official [installation guide](https://github.com/astral-sh/uv).

First, create a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

Then install the project dependencies into your virtual environment:
```bash
uv pip install -e .
```

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
