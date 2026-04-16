"""Tests for fairness metrics."""
# pylint: disable=line-too-long, missing-function-docstring

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pytest
from aif360.sklearn.metrics import (
    average_odds_difference,
    average_predictive_value_difference,
    consistency_score,
    disparate_impact_ratio,
    statistical_parity_difference,
)
from hypothesis import Verbosity, assume, given, settings
from hypothesis import strategies as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.core.metrics.fairness.group.disparate_impact_ratio import DisparateImpactRatio
from src.core.metrics.fairness.group.group_average_odds_difference import (
    GroupAverageOddsDifference,
)
from src.core.metrics.fairness.group.group_average_predictive_value_difference import (
    GroupAveragePredictiveValueDifference,
)
from src.core.metrics.fairness.group.group_statistical_parity_difference import (
    GroupStatisticalParityDifference,
)
from src.core.metrics.fairness.individual.individual_consistency import (
    IndividualConsistency,
)


# generate synthetic bank churn data for testing
def generate_data(n_rows: int = 1000) -> dict[str, np.ndarray]:
    """Generate synthetic bank churn data for testing.

    Args:
        n_rows: Number of rows to generate

    Returns:
        Dictionary mapping column names to numpy arrays

    """
    rng = np.random.default_rng(42)

    return {
        "CreditScore": rng.integers(350, 850, n_rows),
        "Geography": rng.choice(["France", "Germany", "Spain"], n_rows),
        "Gender": rng.choice(["Male", "Female"], n_rows),
        "Age": rng.integers(18, 80, n_rows),
        "Tenure": rng.integers(0, 10, n_rows),
        "Balance": np.round(rng.uniform(0, 200000, n_rows), 2),
        "NumOfProducts": rng.integers(1, 4, n_rows),
        "EstimatedSalary": np.round(rng.uniform(500, 200000, n_rows), 2),
        "Card Type": rng.choice(
            ["SILVER", "GOLD", "PLATINUM", "DIAMOND"],
            n_rows,
        ),
        "Point Earned": rng.integers(200, 1000, n_rows),
        "HasCrCard": rng.integers(0, 2, n_rows),
        "IsActiveMember": rng.integers(0, 2, n_rows),
        "Exited": rng.integers(0, 2, n_rows),
        "Complain": rng.integers(0, 2, n_rows),
        "Satisfaction Score": rng.integers(1, 6, n_rows),
    }


data_dict = generate_data()
df = pd.DataFrame(data_dict)

X = df.drop(columns=["Exited"])
y = df["Exited"]


def train_model(X: pd.DataFrame = X, y: pd.Series = y) -> pd.DataFrame:
    """Train a logistic regression model on encoded and scaled features."""
    categorical_features = [
        "Geography",
        "Gender",
        "Card Type",
        "HasCrCard",
        "IsActiveMember",
        "Complain",
    ]
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        X[feature] = label_encoders[feature].fit_transform(X[feature])  # type: ignore[index]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression().fit(X_scaled, y)

    return pd.DataFrame(lr.predict(X_scaled))


def truth_predict_output(
    X: pd.DataFrame = X,
    y: pd.Series = y,
) -> tuple[pd.Series, pd.DataFrame]:
    """Generate ground truth and predictions indexed by gender for fairness testing."""
    y.index = X["Gender"]
    y_pred = pd.DataFrame(train_model())
    y_pred.index = X["Gender"]
    return y, y_pred


def get_privileged_unprivileged_split(
    df: pd.DataFrame = df,
) -> tuple[np.ndarray, np.ndarray]:
    """Split data into privileged (Male) and unprivileged (Female) groups by gender."""
    data_df = df[[col for col in df.columns if col != "Exited"] + ["Exited"]]
    data = data_df.to_numpy()
    privileged = data[np.where(data[:, 2] == "Male")]
    unprivileged = data[np.where(data[:, 2] == "Female")]
    return privileged, unprivileged


def get_labeled_data(df: pd.DataFrame = df) -> tuple[np.ndarray, np.ndarray]:
    """Generate ground truth and predicted data arrays for fairness testing."""
    data_df = df[[col for col in df.columns if col != "Exited"] + ["Exited"]]
    data = data_df.to_numpy()
    y_pred = pd.DataFrame(train_model())
    data_pred = data.copy()
    data_pred[:, -1] = y_pred.to_numpy().flatten()
    return data, data_pred


def get_k_neighbors_function(k_value: int = 5) -> Callable:
    """Create a function that returns k nearest neighbors for a given input."""

    def find_neighbors(sample: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """Find k nearest neighbors for a given sample."""
        if isinstance(sample, np.ndarray) and sample.ndim > 1:
            sample = sample.flatten()

        nbrs = NearestNeighbors(n_neighbors=k_value + 1, algorithm="ball_tree").fit(
            samples,
        )
        _, indices = nbrs.kneighbors([sample])

        neighbor_indices = indices[0][1 : k_value + 1]
        return cast("np.ndarray", samples[neighbor_indices])

    return find_neighbors


def get_processed_data(
    X: pd.DataFrame = X,
    sample_size: int | None = None,
) -> np.ndarray:
    """Process data for testing individual consistency."""
    categorical_features = [
        "Geography",
        "Gender",
        "Card Type",
        "HasCrCard",
        "IsActiveMember",
        "Complain",
    ]
    X_processed = X.copy()
    for feature in categorical_features:
        if feature in X_processed.columns:
            le = LabelEncoder()
            X_processed[feature] = le.fit_transform(X_processed[feature])  # type: ignore[index]

    if sample_size is not None:
        return X_processed.to_numpy()[:sample_size]
    return X_processed.to_numpy()


class MockPredictionProvider:
    """Mock prediction provider for testing."""

    def __init__(self, predictions: np.ndarray) -> None:
        """Initialize with precomputed predictions."""
        self.predictions = predictions

    def predict(self, x: np.ndarray) -> list[list[int]]:
        """Return prediction for input."""
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)

        result = []
        for i in range(x.shape[0]):
            if i < len(self.predictions):
                result.append([self.predictions[i][0]])
            else:
                result.append([0])
        return result


class PerfectConsistencyProvider:
    """Provider that always returns the same prediction."""

    def predict(self, x: np.ndarray) -> list[list[int]]:
        """Return consistent predictions (all 1s) for all inputs."""
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)
        return [[1] for _ in range(x.shape[0])]


class RandomPredictionProvider:
    """Provider that returns random predictions."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize random number generator with seed for reproducibility."""
        self.rng = np.random.RandomState(seed)

    def predict(self, x: np.ndarray) -> list[list[int]]:
        """Return random binary predictions for inputs."""
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)
        return [[self.rng.randint(0, 2)] for _ in range(x.shape[0])]


def set_favorable_outcomes(group: np.ndarray, target_rate: float = 0.5) -> None:
    """Set favorable outcomes in a group at a target rate."""
    rng = np.random.default_rng()
    num_favorable = round(len(group) * target_rate)
    group[:, -1] = 0
    if 0 < num_favorable <= len(group):
        indices = rng.choice(len(group), size=num_favorable, replace=False)
        group[indices, -1] = 1


y, y_pred = truth_predict_output()
privileged, unprivileged = get_privileged_unprivileged_split()
data, data_pred = get_labeled_data()


@st.composite
def bank_data_strategy(draw: st.DrawFn) -> pd.DataFrame:
    """Generate synthetic bank data for property-based testing.

    Generates variable row counts for comprehensive test coverage.
    """
    n_rows = draw(st.integers(min_value=10, max_value=1000))
    data = generate_data(n_rows)
    return pd.DataFrame(data)


class TestDisparateImpactRatio:
    """Test suite for Disparate Impact Ratio metric."""

    def test_dir_consistent_with_sklearn(self) -> None:
        """Verify DIR calculation matches scikit-learn AIF360 implementation."""
        dir_result = disparate_impact_ratio(
            y,
            prot_attr="Gender",
            priv_group="Male",
            pos_label=1,
        )

        score = DisparateImpactRatio.calculate(
            privileged=privileged,
            unprivileged=unprivileged,
            favorable_outputs=np.array([1]),
        )
        assert score == pytest.approx(dir_result, abs=1e-5)

    @given(bank_data_strategy())
    @settings(max_examples=20)
    def test_dir_always_positive(self, df: pd.DataFrame) -> None:
        """Property-based test to verify the result of DIR calculation is always positive."""
        privileged, unprivileged = get_privileged_unprivileged_split(df=df)

        # Filter out cases with no favorable outcomes to prevent division by zero
        assume(np.any(privileged[:, -1] == 1))
        assume(np.any(unprivileged[:, -1] == 1))

        score = DisparateImpactRatio.calculate(
            privileged=privileged,
            unprivileged=unprivileged,
            favorable_outputs=np.array([1]),
        )

        assert score >= 0, f"Disparate Impact Ratio {score} should be >= 0"

    def test_dir_zero_favorable_unprivileged(self) -> None:
        """Tests that the result of DIR calculation is zero when unprivileged group has no favorable outcomes."""
        df = pd.DataFrame(generate_data())
        privileged, unprivileged = get_privileged_unprivileged_split(df=df)

        # Modify the data so that the unprivileged group has zero favorable outcomes
        unprivileged[:, -1] = 0

        score = DisparateImpactRatio.calculate(
            privileged=privileged,
            unprivileged=unprivileged,
            favorable_outputs=np.array([1]),
        )

        assert score == 0, (
            f"DIR should be zero when there are no favorable outcomes in the "
            f"unprivileged group. Actual score {score}"
        )

    def test_dir_equal_favorable_rates(self) -> None:
        """Test to check that the result of DIR calculation is 1.0 when favorable outcome rates are equal between groups."""
        df = pd.DataFrame(generate_data())
        privileged, unprivileged = get_privileged_unprivileged_split(df=df)

        # Set both groups to have the same favorable outcome rate
        set_favorable_outcomes(group=privileged)
        set_favorable_outcomes(group=unprivileged)

        score = DisparateImpactRatio.calculate(
            privileged=privileged,
            unprivileged=unprivileged,
            favorable_outputs=np.array([1]),
        )
        assert score == pytest.approx(1.0, abs=1e-5), (
            f"DIR should be ~1 when rates are equal. Actual score: {score}"
        )

    def test_dir_empty_dataframe(self) -> None:
        """Test that DIR calculation handles an empty DataFrame correctly."""
        empty_df = pd.DataFrame(columns=df.columns)
        privileged, unprivileged = get_privileged_unprivileged_split(df=empty_df)

        with pytest.raises(ValueError, match=r"."):
            DisparateImpactRatio.calculate(
                privileged=privileged,
                unprivileged=unprivileged,
                favorable_outputs=np.array([1]),
            )


class TestGroupStatisticalParityDifference:
    """Test suite for Group Statistical Parity Difference metric."""

    def test_spd_consistent_with_sklearn(self) -> None:
        """Verify SPD calculation matches scikit-learn AIF360 implementation."""
        spd = statistical_parity_difference(
            y,
            prot_attr="Gender",
            priv_group="Male",
            pos_label=1,
        )

        score = GroupStatisticalParityDifference.calculate(
            privileged=privileged,
            unprivileged=unprivileged,
            favorable_outputs=np.array([1]),
        )

        assert score == pytest.approx(spd, abs=1e-5)

    @given(bank_data_strategy())
    @settings(max_examples=20, verbosity=Verbosity.normal)
    def test_spd_range(self, df: pd.DataFrame) -> None:
        """Property-based test to verify the result of DIR calculation is always positive."""
        privileged, unprivileged = get_privileged_unprivileged_split(df=df)

        # Filter out cases with no favorable outcomes to prevent division by zero
        assume(np.any(privileged[:, -1] == 1))
        assume(np.any(unprivileged[:, -1] == 1))

        score = GroupStatisticalParityDifference.calculate(
            privileged=privileged,
            unprivileged=unprivileged,
            favorable_outputs=np.array([1]),
        )

        assert -1 <= score <= 1, f"SPD {score}, should be between -1 and 1"

    def test_spd_zero_when_equal_rates(self) -> None:
        """Test that SPD is zero when both groups have the same favorable outcome rate."""
        df = pd.DataFrame(generate_data())
        privileged, unprivileged = get_privileged_unprivileged_split(df=df)

        # Set both groups to have the same favorable outcome rate
        set_favorable_outcomes(group=privileged)
        set_favorable_outcomes(group=unprivileged)

        score = GroupStatisticalParityDifference.calculate(
            privileged=privileged,
            unprivileged=unprivileged,
            favorable_outputs=np.array([1]),
        )
        assert score == pytest.approx(0, abs=1e-2), (
            f"SPD should be ~0 when rates are equal. Actual score: {score}"
        )

    def test_spd_sign(self) -> None:
        """Tests that the sign of SPD correctly indicates which group has a higher favorable outcome rate."""
        df = pd.DataFrame(generate_data())
        privileged, unprivileged = get_privileged_unprivileged_split(df=df)

        # Case 1: Unprivileged group has a higher rate, expecting a positive SPD
        set_favorable_outcomes(group=unprivileged, target_rate=0.8)
        set_favorable_outcomes(group=privileged, target_rate=0.2)

        positive_spd = GroupStatisticalParityDifference.calculate(
            privileged=privileged,
            unprivileged=unprivileged,
            favorable_outputs=np.array([1]),
        )

        assert positive_spd > 0, (
            f"SPD should be positive when unprivileged rate is higher. "
            f"Actual: {positive_spd}"
        )

        # Case 2: Privileged group has a higher rate, expecting a negative SPD
        set_favorable_outcomes(group=unprivileged, target_rate=0.2)
        set_favorable_outcomes(group=privileged, target_rate=0.8)

        negative_spd = GroupStatisticalParityDifference.calculate(
            privileged=privileged,
            unprivileged=unprivileged,
            favorable_outputs=np.array([1]),
        )

        assert negative_spd < 0, (
            f"SPD should be negative when privileged rate is higher. "
            f"Actual: {negative_spd}"
        )

    def test_spd_empty_dataframe(self) -> None:
        """Test that SPD calculation handles an empty DataFrame correctly."""
        empty_df = pd.DataFrame(columns=df.columns)
        privileged, unprivileged = get_privileged_unprivileged_split(df=empty_df)

        with pytest.raises(ValueError, match=r"."):
            GroupStatisticalParityDifference.calculate(
                privileged=privileged,
                unprivileged=unprivileged,
                favorable_outputs=np.array([1]),
            )


def test_average_odds_difference() -> None:
    """Verify Average Odds Difference calculation matches AIF360 implementation."""
    aod = average_odds_difference(
        y,
        y_pred,
        prot_attr="Gender",
        priv_group="Male",
        pos_label=1,
    )

    score = GroupAverageOddsDifference.calculate(
        test=data_pred,
        truth=data,
        privilege_columns=[2],
        privilege_values=["Male"],  # type: ignore[list-item]
        positive_class=1,
        output_column=-1,
    )

    assert score == pytest.approx(aod, abs=1e-5)


def test_average_predictive_value_difference() -> None:
    """Verify APVD calculation matches AIF360 implementation.

    Tests Average Predictive Value Difference metric accuracy.
    """
    apvd = average_predictive_value_difference(
        y,
        y_pred,
        prot_attr="Gender",
        priv_group="Male",
        pos_label=1,
    )

    score = GroupAveragePredictiveValueDifference.calculate(
        test=data_pred,
        truth=data,
        privilege_columns=[2],
        privilege_values=["Male"],  # type: ignore[list-item]
        positive_class=1,
        output_column=-1,
    )

    assert score == pytest.approx(apvd, abs=0.2)


def test_individual_consistency() -> None:
    """Test individual consistency calculation using AIF360's consistency_score as ground truth."""
    X_sample = get_processed_data(sample_size=50)
    y_pred_sample = y_pred.iloc[:50].to_numpy()

    k = 5
    cs_score = consistency_score(X_sample, y_pred_sample.flatten())

    model = MockPredictionProvider(y_pred_sample)
    proximity_function = get_k_neighbors_function(k)

    score = IndividualConsistency.calculate(
        proximity_function=proximity_function,
        samples=X_sample,
        model=model,
    )

    assert score == pytest.approx(cs_score, abs=0.25)


def test_individual_consistency_perfect() -> None:
    """Test individual consistency with a perfect consistency model."""
    X_sample = get_processed_data(sample_size=20)

    perfect_predictions = np.ones(20)

    cs_score = consistency_score(X_sample, perfect_predictions)

    proximity_function = get_k_neighbors_function(3)

    consistency = IndividualConsistency.calculate(
        proximity_function=proximity_function,
        samples=X_sample,
        model=PerfectConsistencyProvider(),
    )

    assert consistency == pytest.approx(cs_score, abs=0.2)


def test_individual_consistency_imperfect() -> None:
    """Test individual consistency with an inconsistent model."""
    X_sample = get_processed_data(sample_size=20)

    rng = np.random.RandomState(42)
    random_predictions = rng.randint(0, 2, size=20)

    cs_score = consistency_score(X_sample, random_predictions)

    proximity_function = get_k_neighbors_function(3)

    consistency = IndividualConsistency.calculate(
        proximity_function=proximity_function,
        samples=X_sample,
        model=RandomPredictionProvider(seed=42),
    )

    assert consistency == pytest.approx(cs_score, abs=0.2)
