"""Unit tests for calculate_batch_mean_metric."""

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trustyai_service.endpoints.metrics.batch_mean import (
    BatchMeanRequest,
    calculate_batch_mean_metric,
)


def _request(column: str = "col") -> BatchMeanRequest:
    """Create a minimal BatchMeanRequest for testing."""
    return BatchMeanRequest(modelId="test", columnName=column)


class TestCalculateBatchMeanMetric:
    """Tests for the core batch mean calculation function."""

    def test_correct_mean(self) -> None:
        """Mean of [2, 4, 6, 8, 10] is 6.0."""
        df = pd.DataFrame({"col": [2.0, 4.0, 6.0, 8.0, 10.0]})
        result = calculate_batch_mean_metric(df, _request())
        assert result.get_value() == 6.0  # noqa: PLR2004

    def test_single_value(self) -> None:
        """Mean of a single value is that value."""
        df = pd.DataFrame({"col": [42.0]})
        result = calculate_batch_mean_metric(df, _request())
        assert result.get_value() == 42.0  # noqa: PLR2004

    def test_negative_values(self) -> None:
        """Mean of symmetric negatives and positives is 0.0."""
        df = pd.DataFrame({"col": [-3.0, -1.0, 1.0, 3.0]})
        result = calculate_batch_mean_metric(df, _request())
        assert result.get_value() == 0.0

    def test_skips_nan_values(self) -> None:
        """NaN values are filtered before computing the mean."""
        df = pd.DataFrame({"col": [1.0, float("nan"), 3.0, float("nan"), 5.0]})
        result = calculate_batch_mean_metric(df, _request())
        assert result.get_value() == 3.0  # noqa: PLR2004

    def test_all_nan_returns_nan(self) -> None:
        """All-NaN column returns NaN."""
        df = pd.DataFrame({"col": [float("nan"), float("nan")]})
        result = calculate_batch_mean_metric(df, _request())
        assert math.isnan(result.get_value())

    def test_missing_column_raises(self) -> None:
        """Requesting a non-existent column raises ValueError."""
        df = pd.DataFrame({"other": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found"):
            calculate_batch_mean_metric(df, _request("nonexistent"))

    def test_large_batch(self) -> None:
        """Large batch mean matches numpy reference value."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal(10_000)
        df = pd.DataFrame({"col": values})
        result = calculate_batch_mean_metric(df, _request())
        assert result.get_value() == pytest.approx(np.mean(values), abs=1e-10)

    def test_integer_column(self) -> None:
        """Integer columns are handled correctly."""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        result = calculate_batch_mean_metric(df, _request())
        assert result.get_value() == 3.0  # noqa: PLR2004

    def test_string_column_raises(self) -> None:
        """String columns raise TypeError with descriptive message."""
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        with pytest.raises(TypeError, match="non-numeric"):
            calculate_batch_mean_metric(df, _request())


class TestBatchMeanRequestValidation:
    """Tests for Pydantic validation on BatchMeanRequest."""

    def test_batch_size_zero_rejected(self) -> None:
        """batch_size=0 is rejected by ge=1 constraint."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            BatchMeanRequest(modelId="test", columnName="col", batchSize=0)

    def test_batch_size_negative_rejected(self) -> None:
        """Negative batch_size is rejected by ge=1 constraint."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            BatchMeanRequest(modelId="test", columnName="col", batchSize=-1)
