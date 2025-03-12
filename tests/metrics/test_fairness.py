# pylint: disable=line-too-long, missing-function-docstring
from typing import List, Optional

from pytest import approx
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from aif360.sklearn.metrics import (
    disparate_impact_ratio,
    statistical_parity_difference,
    average_odds_difference,
    average_predictive_value_difference,
)

from src.core.metrics.fairness.group.disparate_impact_ratio import DisparateImpactRatio
from src.core.metrics.fairness.group.group_average_odds_difference import GroupAverageOddsDifference
from src.core.metrics.fairness.group.group_average_predictive_value_difference import GroupAveragePredictiveValueDifference
from src.core.metrics.fairness.group.group_statistical_parity_difference import GroupStatisticalParityDifference

df = pd.read_csv(
    "https://raw.githubusercontent.com/trustyai-explainability/model-collection/8aa8e2e762c6d2b41dbcbe8a0035d50aa5f58c93/bank-churn/data/train.csv",
)
X = df.drop(columns=["Exited"], axis=1)
y = df["Exited"]

def train_model():
    categorical_features = ['Geography', 'Gender', 'Card Type', 'HasCrCard', 'IsActiveMember', 'Complain']
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        X[feature] = label_encoders[feature].fit_transform(X[feature])
    lr = LogisticRegression().fit(X, y)

    y_pred = pd.DataFrame(lr.predict(X))
    return y_pred

def truth_predict_output():
    y.index = X["Gender"]
    y_pred = pd.DataFrame(train_model())
    y_pred.index = X["Gender"]
    return y, y_pred

def get_privileged_unprivleged_split():
    data = df[[col for col in df.columns if col != "Exited"] + ["Exited"]]
    data = data.to_numpy()
    privileged = data[np.where(data[:, 2] == "Male")]
    unprivileged = data[np.where(data[:, 2] == "Female")]
    return privileged, unprivileged

def get_labeled_data():
    data = df[[col for col in df.columns if col != "Exited"] + ["Exited"]]
    data = data.to_numpy()
    y_pred = pd.DataFrame(train_model())
    data_pred = data.copy()
    data_pred[:, -1] = y_pred.to_numpy().flatten()
    return data, data_pred

y, y_pred = truth_predict_output()
privileged, unprivileged = get_privileged_unprivleged_split()
data, data_pred = get_labeled_data()


def test_disparate_impact_ratio():
    dir = disparate_impact_ratio(y, prot_attr="Gender", priv_group="Male", pos_label=1)

    score = DisparateImpactRatio.calculate(
        privileged=privileged,
        unprivileged=unprivileged,
        favorable_output=1
    )
    assert score == approx(dir, abs=1e-5)


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

    assert score == approx(aod, abs=0.2)


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
