"""Stage 4: train LightGBM with expanding-window TS-CV; log to MLflow."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.utils import ROOT, load_params

try:
    import mlflow
except ImportError:  # pragma: no cover
    class _NoOpMLflow:
        """Fallback so the pipeline runs without MLflow installed."""

        def set_experiment(self, *a, **k): pass
        def start_run(self, *a, **k):
            class _Ctx:
                def __enter__(self): return self
                def __exit__(self, *a): return False
            return _Ctx()
        def log_params(self, *a, **k): pass
        def log_param(self, *a, **k): pass
        def log_metric(self, *a, **k): pass
        def log_artifact(self, *a, **k): pass

    mlflow = _NoOpMLflow()

DT = "Datetime"


def feature_cols(df, target):
    return [c for c in df.columns if c not in (DT, target)]


def split_holdout(df, frac):
    n_test = int(len(df) * frac)
    return df.iloc[:-n_test], df.iloc[-n_test:]


def main() -> None:
    p = load_params()
    target = p["data"]["target_col"]
    df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    train_df, _ = split_holdout(df, p["train"]["test_year_fraction"])

    cols = feature_cols(df, target)
    X, y = train_df[cols], train_df[target]

    mcfg = dict(p["train"]["model"])
    if p["train"].get("smoke_test"):
        mcfg["n_estimators"] = 40  # fast path for CI

    mlflow.set_experiment("aep-load-forecasting")
    with mlflow.start_run():
        mlflow.log_params(mcfg)
        mlflow.log_param("n_features", len(cols))

        tscv = TimeSeriesSplit(n_splits=p["train"]["cv_splits"])
        fold_maes = []
        for k, (tr, va) in enumerate(tscv.split(X)):
            m = lgb.LGBMRegressor(**mcfg)
            m.fit(X.iloc[tr], y.iloc[tr])
            mae = mean_absolute_error(y.iloc[va], m.predict(X.iloc[va]))
            fold_maes.append(mae)
            mlflow.log_metric("cv_mae", mae, step=k)

        mlflow.log_metric("cv_mae_mean", float(np.mean(fold_maes)))
        mlflow.log_metric("cv_mae_std", float(np.std(fold_maes)))

        # Refit on full training portion, persist model + feature list.
        final = lgb.LGBMRegressor(**mcfg)
        final.fit(X, y)
        mdir = ROOT / "models"
        mdir.mkdir(exist_ok=True)
        final.booster_.save_model(str(mdir / "model.txt"))
        (mdir / "features.json").write_text(json.dumps(cols))
        mlflow.log_artifact(str(mdir / "model.txt"))

        print(f"cv_mae_mean={np.mean(fold_maes):.1f}")


if __name__ == "__main__":
    main()
