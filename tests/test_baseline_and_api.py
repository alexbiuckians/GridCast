"""Tests for the seasonal-naive baseline and the inference API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.baseline import baseline_metrics, seasonal_naive_pred


def _series(n=400):
    idx = pd.date_range("2010-01-01", periods=n, freq="h")
    # Weekly-periodic signal so seasonal-naive should do well.
    vals = 10000 + 500 * np.sin(2 * np.pi * np.arange(n) / 168)
    return pd.Series(vals, index=idx)


def test_seasonal_naive_shift():
    s = _series(300)
    pred = seasonal_naive_pred(s, season=168)
    # Prediction at t should equal actual at t-168.
    assert pred.iloc[200] == pytest.approx(s.iloc[200 - 168])


def test_baseline_metrics_keys_and_low_error():
    s = _series(400)
    m = baseline_metrics(s, season=168)
    assert set(m) == {"mae", "rmse", "mape_pct"}
    # On a purely weekly-periodic series, seasonal-naive is near-perfect.
    assert m["mape_pct"] < 1.0


def test_health_endpoint():
    from src.serve import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_requires_model_or_validates(tmp_path):
    from src import serve

    client = TestClient(serve.app)
    # If no model is loaded, expect a clear 503; if loaded, a missing-feature 422.
    r = client.post("/predict", json={"features": {}})
    assert r.status_code in (422, 503)
