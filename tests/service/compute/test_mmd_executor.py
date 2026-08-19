"""Tests for the MMD process-pool executor."""

from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import MagicMock, patch

import pytest

from trustyai_service.service.compute import mmd_executor
from trustyai_service.service.compute.mmd_executor import (
    _max_workers,
    reset_mmd_executor,
    run_in_mmd_executor,
    shutdown_mmd_executor,
    start_mmd_executor,
)

from . import _crash_helpers

MODULE_PATH = "trustyai_service.service.compute.mmd_executor"
DEFAULT_WORKERS = 2


@pytest.fixture(autouse=True)
def _reset_executor() -> Generator[None]:
    """Ensure each test starts with no pool and tears down cleanly."""
    reset_mmd_executor()
    yield
    shutdown_mmd_executor()
    reset_mmd_executor()


class TestMaxWorkers:
    """Tests for _max_workers()."""

    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Defaults to DEFAULT_WORKERS when the env var is unset."""
        monkeypatch.delenv("TRUSTYAI_MMD_MAX_WORKERS", raising=False)
        assert _max_workers() == DEFAULT_WORKERS

    def test_valid_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uses the env var value when it's a valid positive int."""
        monkeypatch.setenv("TRUSTYAI_MMD_MAX_WORKERS", "5")
        assert _max_workers() == 5  # noqa: PLR2004

    def test_invalid_env_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to the default for a non-integer env var."""
        monkeypatch.setenv("TRUSTYAI_MMD_MAX_WORKERS", "not-a-number")
        assert _max_workers() == DEFAULT_WORKERS

    def test_below_one_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to the default for a non-positive env var."""
        monkeypatch.setenv("TRUSTYAI_MMD_MAX_WORKERS", "0")
        assert _max_workers() == DEFAULT_WORKERS


class TestStartAndShutdown:
    """Tests for start_mmd_executor() / shutdown_mmd_executor()."""

    def test_start_is_idempotent(self) -> None:
        """Calling start twice only creates one pool."""
        with patch(f"{MODULE_PATH}.ProcessPoolExecutor") as mock_executor_cls:
            mock_executor_cls.return_value = MagicMock()
            start_mmd_executor()
            start_mmd_executor()
            assert mock_executor_cls.call_count == 1

    def test_start_uses_spawn_context_and_initializer(self) -> None:
        """The pool is created with a spawn context and the goodpoints guard."""
        with patch(f"{MODULE_PATH}.ProcessPoolExecutor") as mock_executor_cls:
            mock_executor_cls.return_value = MagicMock()
            start_mmd_executor()
            _, kwargs = mock_executor_cls.call_args
            assert kwargs["mp_context"].get_start_method() == "spawn"
            assert kwargs["initializer"] is mmd_executor.apply_goodpoints_patches

    def test_shutdown_is_idempotent_and_resets_state(self) -> None:
        """Calling shutdown twice only shuts the pool down once."""
        mock_pool = MagicMock()
        with patch(f"{MODULE_PATH}.ProcessPoolExecutor", return_value=mock_pool):
            start_mmd_executor()
        shutdown_mmd_executor()
        shutdown_mmd_executor()
        mock_pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


@pytest.mark.asyncio
class TestRunInMmdExecutor:
    """Tests for run_in_mmd_executor()."""

    async def test_raises_without_start(self) -> None:
        """Using the executor before start_mmd_executor() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            await run_in_mmd_executor(_crash_helpers.add_one, x=1)

    async def test_runs_function_with_kwargs(self) -> None:
        """The submitted function runs in the pool and returns its result."""
        with patch(
            f"{MODULE_PATH}.ProcessPoolExecutor",
            return_value=ProcessPoolExecutor(max_workers=1),
        ):
            start_mmd_executor()
        result = await run_in_mmd_executor(_crash_helpers.add_one, x=41)
        assert result == 42  # noqa: PLR2004


@pytest.mark.asyncio
class TestWorkerCrashIsolation:
    """Proves a crashed worker surfaces as BrokenProcessPool, not a dead test process."""

    async def test_worker_crash_surfaces_as_broken_process_pool(self) -> None:
        """A worker that segfaults breaks only the pool, not the test process."""
        with patch(
            f"{MODULE_PATH}.ProcessPoolExecutor",
            return_value=ProcessPoolExecutor(max_workers=1),
        ):
            start_mmd_executor()
        with pytest.raises(BrokenProcessPool):
            await run_in_mmd_executor(_crash_helpers.die_hard)
