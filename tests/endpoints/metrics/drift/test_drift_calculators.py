"""Tests for drift metric calculator functions used by the Prometheus scheduler."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trustyai_service.endpoints.metrics.drift.compare_means import (
    calculate_compare_means_metric,
)
from trustyai_service.endpoints.metrics.drift.jensen_shannon import (
    calculate_jensenshannon_metric,
)
from trustyai_service.endpoints.metrics.drift.kolmogorov_smirnov import (
    calculate_kstest_metric,
)


def _make_request(
    model_id: str = "test-model",
    reference_tag: str = "TRAINING",
    fit_columns: list[str] | None = None,
) -> MagicMock:
    """Create a mock metric request."""
    request = MagicMock()
    request.model_id = model_id
    request.reference_tag = reference_tag
    request.fit_columns = fit_columns or []
    request.alpha = 0.05
    request.equal_var = True
    request.nan_policy = "omit"
    request.threshold_delta = 0.05
    request.statistic = "distance"
    request.threshold = 0.1
    request.method = "kde"
    request.grid_points = 100
    request.bins = 10
    return request


def _make_dataframe(n_rows: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n_rows),
            "feature2": rng.normal(5, 2, n_rows),
        }
    )


class TestCompareMeansCalculator:
    """Tests for calculate_compare_means_metric."""

    @patch("trustyai_service.endpoints.metrics.drift.compare_means.get_data_source")
    @pytest.mark.asyncio
    async def test_returns_named_values(self, mock_get_ds: MagicMock) -> None:
        """Calculator returns MetricValueCarrier with per-feature statistics."""
        ref_df = _make_dataframe()
        cur_df = _make_dataframe()
        mock_ds = MagicMock()
        mock_ds.get_dataframe_by_tag = AsyncMock(return_value=ref_df)
        mock_get_ds.return_value = mock_ds

        request = _make_request(fit_columns=["feature1", "feature2"])
        result = await calculate_compare_means_metric(cur_df, request)

        assert not result.is_single()
        named = result.get_named_values()
        assert "feature1" in named
        assert "feature2" in named
        assert isinstance(named["feature1"], float)

    @patch("trustyai_service.endpoints.metrics.drift.compare_means.get_data_source")
    @pytest.mark.asyncio
    async def test_derives_columns_from_batch(self, mock_get_ds: MagicMock) -> None:
        """When fit_columns is empty, uses batch columns."""
        ref_df = _make_dataframe()
        cur_df = _make_dataframe()
        mock_ds = MagicMock()
        mock_ds.get_dataframe_by_tag = AsyncMock(return_value=ref_df)
        mock_get_ds.return_value = mock_ds

        request = _make_request(fit_columns=[])
        result = await calculate_compare_means_metric(cur_df, request)

        named = result.get_named_values()
        assert set(named.keys()) == {"feature1", "feature2"}


class TestKSTestCalculator:
    """Tests for calculate_kstest_metric."""

    @patch("trustyai_service.endpoints.metrics.drift.kolmogorov_smirnov.get_data_source")
    @pytest.mark.asyncio
    async def test_returns_named_values(self, mock_get_ds: MagicMock) -> None:
        """Calculator returns per-feature KS statistics."""
        ref_df = _make_dataframe()
        cur_df = _make_dataframe()
        mock_ds = MagicMock()
        mock_ds.get_dataframe_by_tag = AsyncMock(return_value=ref_df)
        mock_get_ds.return_value = mock_ds

        request = _make_request(fit_columns=["feature1"])
        result = await calculate_kstest_metric(cur_df, request)

        assert not result.is_single()
        named = result.get_named_values()
        assert "feature1" in named


class TestJensenShannonCalculator:
    """Tests for calculate_jensenshannon_metric."""

    @patch("trustyai_service.endpoints.metrics.drift.jensen_shannon.get_data_source")
    @pytest.mark.asyncio
    async def test_returns_named_values(self, mock_get_ds: MagicMock) -> None:
        """Calculator returns per-feature JS distance values."""
        ref_df = _make_dataframe()
        cur_df = _make_dataframe()
        mock_ds = MagicMock()
        mock_ds.get_dataframe_by_tag = AsyncMock(return_value=ref_df)
        mock_get_ds.return_value = mock_ds

        request = _make_request(fit_columns=["feature1"])
        result = await calculate_jensenshannon_metric(cur_df, request)

        assert not result.is_single()
        named = result.get_named_values()
        assert "feature1" in named
        assert named["feature1"] >= 0
