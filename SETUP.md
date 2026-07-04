# Setup & Quickstart

## 1. Local environment

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the pipeline

```bash
python -m src.ingest
python -m src.clean
python -m src.features
python -m src.train
python -m src.evaluate
cat metrics.json
```

Expected: model **~2.7% MAPE** vs seasonal-naive **~9.5% MAPE**, skill score **~0.72**.
Plots are written to `reports/figures/`.

## 3. Initialize DVC

```bash
git init && dvc init
dvc add data/raw/AEP_hourly.csv
git add data/raw/AEP_hourly.csv.dvc data/raw/.gitignore .dvc
dvc repro
dvc metrics show
```

Optional remote (off-machine data versioning):
```bash
dvc remote add -d storage s3://your-bucket/dvc   # or gdrive://, azure://, etc.
dvc push
```

## 4. Experiment tracking

```bash
mlflow ui      # http://localhost:5000
```

Each `train` run logs hyperparameters, per-fold CV MAE, mean/std CV MAE, and the model
artifact. Edit `params.yaml` (lags, windows, model hyperparameters) and re-run
`python -m src.train` to generate comparable runs.

## 5. Serve the model

```bash
uvicorn src.serve:app --reload
curl http://localhost:8000/health
```

Docker:
```bash
docker build -t aep-load-api .
docker run -p 8000:8000 aep-load-api
```

## 6. Tests & lint (what CI runs)

```bash
ruff check src tests
pytest -q
```

## 7. Push to GitHub

```bash
git add .
git commit -m "Energy load forecasting MLOps system"
git branch -M main
git remote add origin https://github.com/<you>/energy-load-forecasting.git
git push -u origin main
```

GitHub Actions then runs lint + tests + a training smoke-test + an API smoke-test on every push.

## Experiment ideas (to populate MLflow with comparable runs)

- Toggle `features.cyclical`; add/remove lags or rolling windows.
- Sweep `learning_rate`, `num_leaves`, `n_estimators`.
- Each variation is a tracked, comparable MLflow run.
