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
def test_predict_from_history_returns_prediction():
    """The history endpoint should engineer features server-side and predict."""
    from src.serve import app

    # Enough contiguous hourly history to fill the longest lag/rolling window (168).
    history = [15000 + 500 * (i % 24) for i in range(200)]
    body = {"timestamp": "2018-08-03T14:00:00", "history": history}

    with TestClient(app) as client:  # context manager -> lifespan loads the model
        r = client.post("/predict_from_history", json=body)

    # If the model is present (as in CI after training), expect a real prediction.
    # If artifacts are missing, a clear 503 is acceptable.
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert "predicted_mw" in r.json()
        assert isinstance(r.json()["predicted_mw"], (int, float))


def test_predict_from_history_rejects_short_history():
    """Too little history to build features should be a clear 422, not a crash."""
    from src.serve import app

    body = {"timestamp": "2018-08-03T14:00:00", "history": [15000, 15500, 16000]}
    with TestClient(app) as client:
        r = client.post("/predict_from_history", json=body)
    assert r.status_code == 422