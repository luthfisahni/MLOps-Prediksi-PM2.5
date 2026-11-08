import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from train import prepare_data


def test_dataset_exists():
    path = (
        Path(__file__).resolve().parents[1]
        / "data/processed/pm25_stt_satyabhakti_clean.csv"
    )

    assert path.exists()


def test_prepare_data():
    X_train, X_test, y_train, y_test = prepare_data()

    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty


def test_feature_columns():
    X_train, X_test, _, _ = prepare_data()

    expected_columns = ["lag_1", "lag_2", "lag_3"]

    assert list(X_train.columns) == expected_columns
    assert list(X_test.columns) == expected_columns


def test_no_missing_values():
    X_train, X_test, y_train, y_test = prepare_data()

    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()
    assert not y_train.isnull().values.any()
    assert not y_test.isnull().values.any()


def test_chronological_split():
    X_train, X_test, _, _ = prepare_data()

    assert len(X_train) > len(X_test)