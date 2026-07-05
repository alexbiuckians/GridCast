"""Inference API: serves load forecasts from the trained LightGBM model.

Run: uvicorn src.serve:app --reload

Endpoints:
  GET  /health                -> liveness + model-loaded flag
  POST /predict               -> {"features": {...}} raw engineered features in
  POST /predict_from_history  -> {"timestamp": ..., "history": [...]} the API
                                 engineers features itself, using the same
                                 build() as the training pipeline (no skew).
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import build
from src.utils import ROOT, load_params

_MODEL = None
_FEATURES = None

DT = "Datetime"


def _load():
    """Load model + feature list from disk into module globals."""
    global _MODEL, _FEATURES
    mpath = ROOT / "models" / "model.txt"
    fpath = ROOT / "models" / "features.json"
    if not mpath.exists() or not fpath.exists():
        raise RuntimeError("Model artifacts missing. Run the training pipeline first.")
    _MODEL = lgb.Booster(model_file=str(mpath))
    _FEATURES = json.loads(Path(fpath).read_text())


def _get_model():
    """Lazily load the model on first use.

    This makes the API robust regardless of whether the lifespan startup hook
    ran (e.g. bare TestClient usage), and returns a clear 503 if artifacts are
    still missing.
    """
    if _MODEL is None:
        try:
            _load()
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    return _MODEL, _FEATURES


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort eager load; _get_model() will retry lazily if this no-ops.
    try:
        _load()
    except RuntimeError:
        pass
    yield


app = FastAPI(title="AEP Load Forecasting API", version="1.1", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: dict  # mapping of feature_name -> value


class PredictResponse(BaseModel):
    predicted_mw: float


class HistoryRequest(BaseModel):
    timestamp: str = Field(..., description="ISO datetime to forecast, e.g. 2018-08-03T14:00:00")
    history: list[float] = Field(
        ...,
        description=(
            "Recent hourly AEP_MW values ending at the hour BEFORE `timestamp`, "
            "oldest first. Provide enough history to cover the longest lag/rolling "
            "window (>=168 hours recommended)."
        ),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict from already-engineered features (advanced / debugging use)."""
    _, feats = _get_model()
    missing = [f for f in feats if f not in req.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    row = [[req.features[f] for f in feats]]
    pred = float(_MODEL.predict(row)[0])
    return PredictResponse(predicted_mw=pred)


@app.post("/predict_from_history", response_model=PredictResponse)
def predict_from_history(req: HistoryRequest):
    """Predict from a timestamp + recent load history.

    The API engineers features here using the SAME build() as the training
    pipeline, so training and serving share one feature-construction code path.
    """
    model, feats = _get_model()
    p = load_params()

    try:
        ts = pd.to_datetime(req.timestamp)
    except (ValueError, TypeError):
        raise HTTPException(422, f"Could not parse timestamp: {req.timestamp!r}")

    lags = p["features"]["lags"]
    windows = p["features"]["rolling_windows"]
    cyclical = p["features"].get("cyclical", True)
    target = p["data"]["target_col"]

    # Need enough history to fill the longest lag / rolling window.
    need = max(max(lags), max(windows))
    if len(req.history) < need:
        raise HTTPException(
            422,
            f"Need at least {need} history values to compute features; "
            f"got {len(req.history)}.",
        )

    # Reconstruct an hourly frame ending at the hour before `ts`, then append the
    # target hour (value unknown -> NaN) so build() can compute its feature row.
    n = len(req.history)
    hist_index = pd.date_range(end=ts - pd.Timedelta(hours=1), periods=n, freq="h")
    # Placeholder for the target hour's own value: it is NEVER used as a feature
    # (only lags/rolls of PRIOR hours are), but a NaN here would make build()'s
    # dropna() discard the row we need. Use the last known value as a filler.
    placeholder = float(req.history[-1])
    frame = pd.DataFrame({DT: list(hist_index) + [ts],
                          target: list(req.history) + [placeholder]})

    feat_df = build(frame, DT, target, lags, windows, cyclical=cyclical)
    # The final engineered row corresponds to `ts`. build() drops NaN rows, so
    # the target hour survives only because its features come from history.
    row = feat_df[feat_df[DT] == ts]
    if row.empty:
        raise HTTPException(
            422,
            "Feature construction produced no row for the requested timestamp; "
            "check that history is contiguous hourly data ending just before it.",
        )

    x = [[float(row.iloc[0][f]) for f in feats]]
    pred = float(model.predict(x)[0])
    return PredictResponse(predicted_mw=pred)