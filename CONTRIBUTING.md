# New contributor guide

## Installation

TrustyAIService uses `uv` as a package manager.
To install it, follow the official
[installation guide](https://github.com/astral-sh/uv).

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

We use `pre-commit` with `ruff` and `pyrefly` for code formatting
and type checking. Make sure all pre-commit checks pass before committing.

To install pre-commit:

```bash
pre-commit install -t pre-commit -t commit-msg
```

Run manually pre-commit:

```bash
pre-commit run --all-files
```
