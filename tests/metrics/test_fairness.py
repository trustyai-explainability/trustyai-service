# pylint: disable=line-too-long, missing-function-docstring
from typing import List, Optional

from pytest import approx
import numpy as np
import pandas as pd
import os
import pathlib

from src.core.metrics.fairness.group.disparate_impact_ratio import DisparateImpactRatio
from src.core.metrics.fairness.group.group_average_odds_difference import GroupAverageOddsDifference
from src.core.metrics.fairness.group.group_average_predictive_value_difference import GroupAveragePredictiveValueDifference
from src.core.metrics.fairness.group.group_statistical_parity_difference import GroupStatisticalParityDifference

TEST_DIR = pathlib.Path(__file__).parent.resolve()

INCOME_DF_BIASED = pd.read_csv(os.path.join(TEST_DIR, "data/income-biased.zip"), index_col=False)
INCOME_DF_UNBIASED = pd.read_csv(os.path.join(TEST_DIR, "data/income-unbiased.zip"), index_col=False)

AIF_DF = pd.read_csv(os.path.join(TEST_DIR, "data/data.csv"), index_col=False)

def create_random_dataframe(weights: Optional[List[float]] = None) -> pd.DataFrame:
    """Create a simple random Pandas dataframe"""
    from sklearn.datasets import make_classification
    if not weights:
        weights = [0.9, 0.1]

    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=2, class_sep=2, flip_y=0, weights=weights,
                               random_state=23)

    return pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'y': y
    })


def test_statistical_parity_difference_random():
    """Test Statistical Parity Difference (unbalanced random data)"""
    data = create_random_dataframe().to_numpy()

    privileged = data[np.where(data[:, 0] < 0)]
    unprivileged = data[np.where(data[:, 0] >= 0)]
    score = GroupStatisticalParityDifference.calculate(privileged=privileged,
                                                       unprivileged=unprivileged,
                                                       favorable_output=1,
                                                    )
    assert score == approx(0.9, abs=0.09)


def test_statistical_parity_difference_income():
    """Test Statistical Parity Difference (income data)"""
    data = INCOME_DF_BIASED.copy().to_numpy()

    privileged = data[np.where(data[:, 2] == 1)]
    unprivileged = data[np.where(data[:, 2] == 0)]
    favorable = 1
    score = GroupStatisticalParityDifference.calculate(
        privileged=privileged,
        unprivileged=unprivileged,
        favorable_output=favorable
        )
    assert score == approx(-0.15, abs=0.01)


def test_statisical_parity_difference_AIF():
    """Test Statistical Parity Difference (AIF data)"""
    data = AIF_DF.copy().to_numpy()

    privileged = data[np.where(data[:, 9] == 1)]
    unprivileged = data[np.where(data[:, 9] == 0)]
    favorable = 0
    score = GroupStatisticalParityDifference.calculate(privileged=privileged,
                                                       unprivileged=unprivileged,
                                                       favorable_output=favorable)
    assert score == approx(0.19643287553870947, abs=1e-5)


def test_disparate_impact_ratio_random():
    data = create_random_dataframe(weights=[0.5, 0.5]).to_numpy()

    privileged = data[np.where(data[:, 0] < 0)]
    unprivileged = data[np.where(data[:, 0] >= 0)]
    score = DisparateImpactRatio.calculate(privileged=privileged,
                                           unprivileged=unprivileged,
                                           favorable_output=[1])
    assert score == approx(130.0, abs=5.0)


def test_disparate_impact_ratio_income():
    """Test Disparate Impact Ratio (income data)"""
    data = INCOME_DF_BIASED.copy().to_numpy()

    privileged = data[np.where(data[:, 2] == 1)]
    unprivileged = data[np.where(data[:, 2] == 0)]
    score = DisparateImpactRatio.calculate(privileged, unprivileged, [1])
    assert score == approx(0.4, abs=0.05)


def test_disparate_impact_ratio_AIF():
    """Test Disparate Impact Ratio (AIF data)"""
    data = AIF_DF.copy().to_numpy()

    privileged = data[np.where(data[:, 9] == 1)]
    unprivileged = data[np.where(data[:, 9] == 0)]

    score = DisparateImpactRatio.calculate(privileged=privileged,
                                           unprivileged=unprivileged,
                                           favorable_output=[0])
    assert score == approx(1.28, abs=1e-2)


def test_average_odds_difference():
    """Test Average Odds Difference (unbalanced random data)"""
    PRIVILEGED_CLASS_GENDER = 1
    UNPRIVILEGED_CLASS_GENDER = 0
    PRIVILEGED_CLASS_RACE = 4
    UNPRIVILEGED_CLASS_RACE = 2

    data_biased = INCOME_DF_BIASED.to_numpy()
    data_unbiased = INCOME_DF_UNBIASED.to_numpy()

    score = GroupAverageOddsDifference.calculate(test=data_biased,
                                                 truth=data_unbiased,
                                                 privilege_columns=[1, 2],
                                                 privilege_values=[PRIVILEGED_CLASS_RACE, PRIVILEGED_CLASS_GENDER],
                                                 positive_class=1,
                                                 output_column=3)
    assert score == approx(0.12, abs=0.2)

    score = GroupAverageOddsDifference.calculate(test=data_biased,
                                                truth=data_unbiased,
                                                privilege_columns=[1, 2],
                                                privilege_values=[UNPRIVILEGED_CLASS_RACE, UNPRIVILEGED_CLASS_GENDER],
                                                positive_class=1,
                                                output_column=3)
    assert score == approx(0.2, abs=0.2)


def test_average_predictive_value_difference():
    """Test Average Predictive Value Difference (unbalanced random data)"""
    PRIVILEGED_CLASS_GENDER = 1
    UNPRIVILEGED_CLASS_GENDER = 0
    PRIVILEGED_CLASS_RACE = 4
    UNPRIVILEGED_CLASS_RACE = 2

    data_biased = INCOME_DF_BIASED.to_numpy()
    data_unbiased = INCOME_DF_UNBIASED.to_numpy()

    score = GroupAveragePredictiveValueDifference.calculate(test=data_biased,
                                                            truth=data_unbiased,
                                                            privilege_columns=[1, 2],
                                                            privilege_values=[PRIVILEGED_CLASS_RACE, PRIVILEGED_CLASS_GENDER],
                                                            positive_class=1,
                                                            output_column=3)
    assert score == approx(-0.3, abs=0.3)

    score = GroupAveragePredictiveValueDifference.calculate(test=data_biased,
                                                            truth=data_unbiased,
                                                            privilege_columns=[1, 2],
                                                            privilege_values=[UNPRIVILEGED_CLASS_RACE, UNPRIVILEGED_CLASS_GENDER],
                                                            positive_class=1,
                                                            output_column=3)
    assert score == approx(-0.22, abs=0.3)
