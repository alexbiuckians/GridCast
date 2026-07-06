# GridCast — Hourly Electricity Load Forecasting (MLOps)

> Hourly electricity demand forecasting that beats a seasonal-naive baseline by 72% (2.7% MAPE) — reproducible MLOps pipeline with MLflow and a feature-engineering FastAPI service.

End-to-end, **reproducible, served** machine-learning system that forecasts hourly electricity demand (megawatts) for the AEP region of the PJM grid. The emphasis is **production engineering**: experiment tracking, a reproducible multi-stage pipeline, an honest baseline, automated tests, CI, and a deployable API that performs its own feature engineering so training and serving share one code path.

| Concern | Tool |
|---|---|
| Pipeline | Multi-stage: ingest → clean → features → train → evaluate |
| Experiment tracking | **MLflow** (params, per-fold metrics, artifacts per run) |
| Modeling | LightGBM, expanding-window time-series cross-validation |
| Honest benchmark | Seasonal-naive baseline + **skill score**|
| Serving | **FastAPI** inference API (server-side feature engineering), containerized with **Docker** |
| Testing | **pytest** (data, features, baseline, both API endpoints) |
| CI | **GitHub Actions** (lint + tests + train smoke-test + API smoke-test) |

## Headline result

On a held-out final period (18,169 hours), the model beats the trivial benchmark decisively:

| Metric | Model (LightGBM) | Seasonal-naive baseline |
|---|---|---|
| MAPE | **2.67%** | 9.54% |
| MAE (MW) | **399** | 1437 |
| RMSE (MW) | **527** | 1902 |

**Skill score vs baseline: 0.72** — the  model reduces mean absolute error by 72% over "same hour, last week." Reporting against a baseline is what makes the headline number meaningful rather than decorative. These figures are reproducible from the pipeline (`python -m src.evaluate` writes them to `metrics.json`).


## Problem

Forecast hourly load from a univariate series (`Datetime`, `AEP_MW`). Short-term load forecasting is an operational function at utilities, ISOs/RTOs, and energy traders. The raw series has two columns; predictive signal is **engineered** from calendar structure (including cyclical encodings and US holidays) and the series' own lagged history — two raw columns become 21 features.


## Pipeline stages 

```
ingest -> clean -> features -> train -> evaluate
```

1. **ingest** — parse timestamps, sort.
2. **clean** — drop duplicate DST timestamps, enforce a continuous hourly grid, bound values.
3. **features** — calendar + cyclical (sin/cos) encodings, US holidays, lag (24/48/168h), rolling mean/std. Two raw columns become 21 features.
4. **train** — LightGBM with expanding-window TS-CV; logs params/metrics/artifact to MLflow.
5. **evaluate** — model vs seasonal-naive baseline, skill score, MAE/RMSE/MAPE, and saved predicted-vs-actual and residual plots.

## Serving
The API exposes two prediction paths. The recommended one takes a **timestamp plus recent load history** and engineers the model's features itself — reusing the exact `build()` function from the training pipeline, so there is no train/serve skew. A lower-level endpoint accepting pre-computed features is also available for debugging.


```bash
uvicorn src.serve:app --reload
```
| Endpoint | Method | Body | Returns |
|---|---|---|---|
| /health | GET | - | liveness + `model_loaded` |
| /predict_from_history | POST | `{"timestamp": ..., "history": [...]}` |  `{"predicted_mw": ...}` |
| /predict | POST | `{"features": {...}}` (pre-engineered)  | `{"predicted_mw": ...}` |


### Example: forecast from recent history

`/predict_from_history` engineers the features server-side. Provide at least 168 hours of
contiguous hourly load ending just before the target hour (oldest first):

```bash
curl -X POST http://localhost:8000/predict_from_history \
  -H "Content-Type: application/json" \
  -d '{
        "timestamp": "2018-08-03T14:00:00",
        "history": [15000, 15500, 16000, "... >=168 hourly values, oldest first ..."]
      }'
```

```json
{ "predicted_mw": 19823.65 }
```


Interactive API docs (Swagger UI) are auto-generated at http://localhost:8000/docs.




Or containerized:

```bash
docker build -t aep-load-api .
docker run -p 8000:8000 aep-load-api
```

## Reproduce

```bash
pip install -r requirements.txt
python -m src.ingest && python -m src.clean && python -m src.features \
  && python -m src.train && python -m src.evaluate
cat metrics.json
```


See `SETUP.md` for full git and MLflow initialization and push instructions.

## Tests & CI
``` bash
ruff check src tests
pytest -q
```


GitHub Actions runs lint, unit tests, a fast end-to-end training smoke-test, and an API
smoke-test (model loads and serves) on every push to `main`.