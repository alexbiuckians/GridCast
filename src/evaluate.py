"""Stage 5: evaluate the trained model on the held-out final period.

Reports model metrics, the seasonal-naive baseline, and the skill score
(fractional MAE improvement over baseline). Saves predicted-vs-actual and
residual plots when params.evaluate.plot is true.
"""
import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.baseline import baseline_metrics
from src.train import split_holdout
from src.utils import ROOT, load_params

DT = "Datetime"


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def save_plots(dt_index, y, yhat, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    # Predicted vs actual — show the last two weeks for readability.
    window = 24 * 14
    plt.figure(figsize=(12, 4))
    plt.plot(dt_index[-window:], y[-window:], label="Actual", linewidth=1)
    plt.plot(dt_index[-window:], yhat[-window:], label="Predicted", linewidth=1)
    plt.title("Predicted vs Actual Load (last 14 days of holdout)")
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pred_vs_actual.png", dpi=120)
    plt.close()

    # Residual histogram.
    resid = y - yhat
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=60)
    plt.title("Residuals (Actual - Predicted)")
    plt.xlabel("MW")
    plt.tight_layout()
    plt.savefig(outdir / "residuals.png", dpi=120)
    plt.close()


def main() -> None:
    p = load_params()
    target = p["data"]["target_col"]
    df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    _, test_df = split_holdout(df, p["train"]["test_year_fraction"])

    cols = json.loads((ROOT / "models" / "features.json").read_text())
    booster = lgb.Booster(model_file=str(ROOT / "models" / "model.txt"))
    yhat = booster.predict(test_df[cols])
    y = test_df[target].values

    model_mae = float(mean_absolute_error(y, yhat))
    base = baseline_metrics(test_df[target].reset_index(drop=True), season=168)
    skill = 1.0 - (model_mae / base["mae"])  # >0 means model beats naive

    metrics = {
        "model": {
            "mae": model_mae,
            "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "mape_pct": mape(y, yhat),
        },
        "baseline_seasonal_naive": base,
        "skill_score_vs_baseline": skill,
        "n_test": int(len(y)),
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    if p.get("evaluate", {}).get("plot", False):
        save_plots(
            pd.to_datetime(test_df[DT].values), y, yhat, ROOT / "reports" / "figures"
        )
        print("plots written to reports/figures/")


if __name__ == "__main__":
    main()
