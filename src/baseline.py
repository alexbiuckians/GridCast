"""Seasonal-naive baseline: predict load = load 168h ago (same hour, last week).

This is the honesty check. A learned model is only worth its complexity if it beats
this trivial benchmark. Reported alongside the model in evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def seasonal_naive_pred(series: pd.Series, season: int = 168) -> pd.Series:
    """Forecast each point as the value `season` hours earlier."""
    return series.shift(season)


def baseline_metrics(y_true: pd.Series, season: int = 168) -> dict:
    yhat = seasonal_naive_pred(y_true, season)
    mask = yhat.notna()
    yt, yp = y_true[mask].values, yhat[mask].values
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mape_pct": float(np.mean(np.abs((yt - yp) / yt)) * 100),
    }
