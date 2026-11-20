"""Tests for resolving the DATABASE_ATTEMPT_MIGRATION startup migration mode."""

from __future__ import annotations

import pytest

from trustyai_service.service.migration_mode import (
    MIGRATION_ENV_VAR,
    MigrationMode,
    get_migration_mode,
)


def test_unset_is_automatic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the variable the legacy copy runs automatically."""
    monkeypatch.delenv(MIGRATION_ENV_VAR, raising=False)
    assert get_migration_mode() is MigrationMode.AUTO


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_is_automatic(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """An empty value is treated like an unset variable."""
    monkeypatch.setenv(MIGRATION_ENV_VAR, value)
    assert get_migration_mode() is MigrationMode.AUTO


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " true "])
def test_truthy_values_enable_migration(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Truthy values explicitly enable the full startup migration."""
    monkeypatch.setenv(MIGRATION_ENV_VAR, value)
    assert get_migration_mode() is MigrationMode.ENABLED


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "garbage"])
def test_other_values_disable_migration(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Any other explicit value is an opt-out, matching the previous behavior."""
    monkeypatch.setenv(MIGRATION_ENV_VAR, value)
    assert get_migration_mode() is MigrationMode.DISABLED
