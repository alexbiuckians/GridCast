"""Inference API: serves load forecasts from the trained LightGBM model.

Run: uvicorn src.serve:app --reload
POST /predict with the engineered feature values; GET /health for liveness.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils import ROOT

_MODEL = None
_FEATURES = None


def _load():
    global _MODEL, _FEATURES
    mpath = ROOT / "models" / "model.txt"
    fpath = ROOT / "models" / "features.json"
    if not mpath.exists() or not fpath.exists():
        raise RuntimeError("Model artifacts missing. Run the training pipeline first.")
    _MODEL = lgb.Booster(model_file=str(mpath))
    _FEATURES = json.loads(Path(fpath).read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load()
    except RuntimeError:
        # Allow the app to boot without a trained model yet; /predict will
        # return a clear error until the pipeline has produced artifacts.
        pass
    yield


app = FastAPI(title="AEP Load Forecasting API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: dict  # mapping of feature_name -> value


class PredictResponse(BaseModel):
    predicted_mw: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _MODEL is None:
        raise HTTPException(503, "Model not loaded. Run the pipeline to produce models/model.txt.")
    missing = [f for f in _FEATURES if f not in req.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    row = [[req.features[f] for f in _FEATURES]]
    pred = float(_MODEL.predict(row)[0])
    return PredictResponse(predicted_mw=pred)
