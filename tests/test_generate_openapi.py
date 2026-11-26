"""The OpenAPI generator must not depend on the caller's deployment config."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_openapi.py"


@pytest.mark.parametrize("batch_size", ["", "not-a-number"])
def test_generator_ignores_an_invalid_batch_size(
    batch_size: str,
    tmp_path: Path,
) -> None:
    """An exported SERVICE_BATCH_SIZE must not break the documented command.

    The scheduler parses SERVICE_BATCH_SIZE while ``main`` is imported, so the
    generator has to pin a known value before that import.
    """
    env = dict(os.environ, SERVICE_BATCH_SIZE=batch_size)

    result = subprocess.run(  # noqa: S603  # fixed argv, no shell
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "is up to date" in result.stdout
