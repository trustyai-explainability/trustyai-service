#!/usr/bin/env python3
"""Generate or verify the committed OpenAPI document (docs/openapi.yaml).

The document is ``app.openapi()`` of the main FastAPI application, generated
with every endpoint feature flag pinned so the output does not depend on the
deployment environment.

Usage (needs Python >=3.12,<3.15 and the ``dev`` group for PyYAML)::

    uv run python scripts/generate_openapi.py          # rewrite docs/openapi.yaml
    uv run python scripts/generate_openapi.py --check  # fail if the file is stale
"""

import argparse
import difflib
import os
import sys
import tempfile
from pathlib import Path

import yaml

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "openapi.yaml"

# Pinned flag values. Explainers stay disabled, matching the shipped defaults.
PINNED_FLAGS: dict[str, bool] = {
    "fairness": True,
    "fairness_spd": True,
    "fairness_dir": True,
    "drift": True,
    "drift_ks_test": True,
    "drift_jensen_shannon": True,
    "drift_compare_means": True,
    "drift_mmd": True,
    "explainer": False,
    "explainer_local": False,
    "explainer_global": False,
}


def render() -> str:
    """Return the OpenAPI document as deterministic YAML."""
    # Must run before trustyai_service is imported: flags are read at import time.
    for name, enabled in PINNED_FLAGS.items():
        os.environ[f"TRUSTYAI_ENABLE_{name.upper()}"] = "true" if enabled else "false"
    # Isolate from the caller's storage configuration: PVC needs no extras.
    os.environ["SERVICE_STORAGE_FORMAT"] = "PVC"

    with tempfile.TemporaryDirectory() as storage_dir:
        os.environ["STORAGE_DATA_FOLDER"] = storage_dir

        from trustyai_service.main import app  # noqa: PLC0415
        from trustyai_service.service.config import feature_flags  # noqa: PLC0415

        if feature_flags.ENDPOINTS != PINNED_FLAGS:
            msg = (
                "PINNED_FLAGS is out of sync with feature_flags.ENDPOINTS: "
                f"{sorted(set(feature_flags.ENDPOINTS) ^ set(PINNED_FLAGS))}"
            )
            raise SystemExit(msg)

        return yaml.safe_dump(
            app.openapi(),
            sort_keys=True,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )


def main() -> int:
    """Write the document, or with ``--check`` verify it is up to date."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed document differs from the generated one",
    )
    args = parser.parse_args()

    generated = render()
    if not args.check:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(generated, encoding="utf-8")
        print(f"Wrote {OUTPUT}")  # noqa: T201
        return 0

    current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
    if current == generated:
        print(f"{OUTPUT.name} is up to date")  # noqa: T201
        return 0
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        generated.splitlines(keepends=True),
        fromfile=f"committed/{OUTPUT.name}",
        tofile="generated",
    )
    sys.stderr.writelines(list(diff)[:200])
    print(  # noqa: T201
        "\nOpenAPI document is stale. Run: uv run python scripts/generate_openapi.py",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
