"""Unit tests on data-cleaning and feature-engineering logic."""
import numpy as np
import pandas as pd

from src.clean import clean
from src.features import build

DT = "Datetime"
VAL = "AEP_MW"


def _series(n=400):
    idx = pd.date_range("2010-01-01", periods=n, freq="h")
    return pd.DataFrame({DT: idx, VAL: np.linspace(10000, 12000, n)})


def test_clean_removes_duplicates():
    df = _series(50)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    out = clean(dup, DT, VAL, 1000, 60000)
    assert out[DT].duplicated().sum() == 0


def test_clean_bounds_values():
    df = _series(50)
    df.loc[0, VAL] = 999999  # out of range
    out = clean(df, DT, VAL, 1000, 60000)
    assert out[VAL].max() <= 60000


def test_clean_enforces_hourly_grid():
    df = _series(50).drop(index=10).reset_index(drop=True)  # punch a gap
    out = clean(df, DT, VAL, 1000, 60000)
    deltas = out[DT].diff().dropna().dt.total_seconds().unique()
    assert set(deltas) == {3600.0}


def test_features_no_nans_and_expected_cols():
    df = _series(400)
    feats = build(df, DT, VAL, [24, 168], [24, 168])
    assert feats.isna().sum().sum() == 0
    for c in ["hour", "dayofweek", "is_weekend", "is_holiday", "lag_24", "rollmean_24"]:
        assert c in feats.columns


def test_lag_feature_correctness():
    df = _series(400)
    feats = build(df, DT, VAL, [24], [24])
    # lag_24 at any row should equal the raw value 24 hours earlier
    merged = df.set_index(DT)[VAL]
    sample = feats.iloc[100]
    expected = merged.loc[sample[DT] - pd.Timedelta(hours=24)]
    assert abs(sample["lag_24"] - expected) < 1e-6
