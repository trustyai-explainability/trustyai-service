# pylint: disable=line-too-long, missing-function-docstring
from pytest import approx
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from aif360.sklearn.metrics import (
    disparate_impact_ratio,
    statistical_parity_difference,
    average_odds_difference,
    average_predictive_value_difference,
    consistency_score
)

from src.core.metrics.fairness.group.disparate_impact_ratio import DisparateImpactRatio
from src.core.metrics.fairness.group.group_average_odds_difference import GroupAverageOddsDifference
from src.core.metrics.fairness.group.group_average_predictive_value_difference import GroupAveragePredictiveValueDifference
from src.core.metrics.fairness.group.group_statistical_parity_difference import GroupStatisticalParityDifference
from src.core.metrics.fairness.individual.individual_consistency import IndividualConsistency

# generate synthetic bank churn data for testing
np.random.seed(42)
n_rows = 1000

data = {
    "CreditScore": np.random.randint(350, 850, n_rows),
    "Geography": np.random.choice(["France", "Germany", "Spain"], n_rows),
    "Gender": np.random.choice(["Male", "Female"], n_rows),
    "Age": np.random.randint(18, 80, n_rows),
    "Tenure": np.random.randint(0, 10, n_rows),
    "Balance": np.round(np.random.uniform(0, 200000, n_rows), 2),
    "NumOfProducts": np.random.randint(1, 4, n_rows),
    "EstimatedSalary": np.round(np.random.uniform(500, 200000, n_rows), 2),
    "Card Type": np.random.choice(["SILVER", "GOLD", "PLATINUM", "DIAMOND"], n_rows),
    "Point Earned": np.random.randint(200, 1000, n_rows),
    "HasCrCard": np.random.randint(0, 2, n_rows),
    "IsActiveMember": np.random.randint(0, 2, n_rows),
    "Exited": np.random.randint(0, 2, n_rows),
    "Complain": np.random.randint(0, 2, n_rows),
    "Satisfaction Score": np.random.randint(1, 6, n_rows),
}

df = pd.DataFrame(data)

X = df.drop(columns=["Exited"], axis=1)
y = df["Exited"]

def train_model(X: pd.DataFrame = X, y: pd.Series = y):
    categorical_features = ['Geography', 'Gender', 'Card Type', 'HasCrCard', 'IsActiveMember', 'Complain']
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        X[feature] = label_encoders[feature].fit_transform(X[feature])
    lr = LogisticRegression().fit(X, y)

    return pd.DataFrame(lr.predict(X))

def truth_predict_output(X: pd.DataFrame = X, y: pd.Series = y):
    y.index = X["Gender"]
    y_pred = pd.DataFrame(train_model())
    y_pred.index = X["Gender"]
    return y, y_pred

def get_privileged_unprivleged_split(df: pd.DataFrame = df):
    data = df[[col for col in df.columns if col != "Exited"] + ["Exited"]]
    data = data.to_numpy()
    privileged = data[np.where(data[:, 2] == "Male")]
    unprivileged = data[np.where(data[:, 2] == "Female")]
    return privileged, unprivileged

def get_labeled_data(df: pd.DataFrame = df):
    data = df[[col for col in df.columns if col != "Exited"] + ["Exited"]]
    data = data.to_numpy()
    y_pred = pd.DataFrame(train_model())
    data_pred = data.copy()
    data_pred[:, -1] = y_pred.to_numpy().flatten()
    return data, data_pred


def get_k_neighbors_function(k_value=5):
    """Create a function that returns k nearest neighbors for a given input."""

    def find_neighbors(sample, samples):
        """Find k nearest neighbors for a given sample."""
        if isinstance(sample, np.ndarray) and sample.ndim > 1:
            sample = sample.flatten()

        nbrs = NearestNeighbors(n_neighbors=k_value + 1, algorithm='ball_tree').fit(samples)
        distances, indices = nbrs.kneighbors([sample])

        neighbor_indices = indices[0][1:k_value + 1]
        return samples[neighbor_indices]

    return find_neighbors


def get_processed_data(X: pd.DataFrame = X, sample_size=None):
    """Process data for testing individual consistency."""
    categorical_features = ['Geography', 'Gender', 'Card Type', 'HasCrCard', 'IsActiveMember', 'Complain']
    X_processed = X.copy()
    for feature in categorical_features:
        if feature in X_processed.columns:
            le = LabelEncoder()
            X_processed[feature] = le.fit_transform(X_processed[feature])

    if sample_size is not None:
        return X_processed.to_numpy()[:sample_size]
    return X_processed.to_numpy()


class MockPredictionProvider:
    """Mock prediction provider for testing."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, x):
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

    def predict(self, x):
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)
        return [[1] for _ in range(x.shape[0])]


class RandomPredictionProvider:
    """Provider that returns random predictions."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def predict(self, x):
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)
        return [[self.rng.randint(0, 2)] for _ in range(x.shape[0])]

y, y_pred = truth_predict_output()
privileged, unprivileged = get_privileged_unprivleged_split()
data, data_pred = get_labeled_data()


def test_disparate_impact_ratio():
    dir_result = disparate_impact_ratio(y, prot_attr="Gender", priv_group="Male", pos_label=1)

    score = DisparateImpactRatio.calculate(
        privileged=privileged,
        unprivileged=unprivileged,
        favorable_output=1
    )
    assert score == approx(dir_result, abs=1e-5)


def test_statistical_parity_difference():
    spd = statistical_parity_difference(y, prot_attr="Gender", priv_group="Male", pos_label=1)

    score = GroupStatisticalParityDifference.calculate(
        privileged=privileged,
        unprivileged=unprivileged,
        favorable_output=1
    )

    assert score == approx(spd, abs=1e-5)


def test_average_odds_difference():
    aod = average_odds_difference(y, y_pred, prot_attr="Gender", priv_group="Male", pos_label=1)

    score = GroupAverageOddsDifference.calculate(
        test=data_pred,
        truth=data,
        privilege_columns=[2],
        privilege_values=["Male"],
        positive_class=1,
        output_column=-1
    )

    assert score == approx(aod, abs=1e-5)


def test_average_predictive_value_difference():
    apvd = average_predictive_value_difference(y, y_pred, prot_attr="Gender", priv_group="Male", pos_label=1)

    score = GroupAveragePredictiveValueDifference.calculate(
        test=data_pred,
        truth=data,
        privilege_columns=[2],
        privilege_values=["Male"],
        positive_class=1,
        output_column=-1
    )

    assert score == approx(apvd, abs=0.2)


def test_individual_consistency():
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
        model=model
    )

    assert score == approx(cs_score, abs=0.2)


def test_individual_consistency_perfect():
    """Test individual consistency with a perfect consistency model."""
    X_sample = get_processed_data(sample_size=20)

    perfect_predictions = np.ones(20)

    cs_score = consistency_score(X_sample, perfect_predictions)

    proximity_function = get_k_neighbors_function(3)

    consistency = IndividualConsistency.calculate(
        proximity_function=proximity_function,
        samples=X_sample,
        model=PerfectConsistencyProvider()
    )

    assert consistency == approx(cs_score, abs=0.2)


def test_individual_consistency_imperfect():
    """Test individual consistency with an inconsistent model."""
    X_sample = get_processed_data(sample_size=20)

    rng = np.random.RandomState(42)
    random_predictions = rng.randint(0, 2, size=20)

    cs_score = consistency_score(X_sample, random_predictions)

    proximity_function = get_k_neighbors_function(3)

    consistency = IndividualConsistency.calculate(
        proximity_function=proximity_function,
        samples=X_sample,
        model=RandomPredictionProvider(seed=42)
    )

    assert consistency == approx(cs_score, abs=0.2)
