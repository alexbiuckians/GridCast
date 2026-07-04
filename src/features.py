"""Stage 3: engineer calendar, lag, and rolling features from the univariate series."""
import holidays
import numpy as np
import pandas as pd

from src.utils import ROOT, load_params


def add_calendar(df: pd.DataFrame, dt: str, cyclical: bool = True) -> pd.DataFrame:
    t = df[dt].dt
    df["hour"] = t.hour
    df["dayofweek"] = t.dayofweek
    df["month"] = t.month
    df["dayofyear"] = t.dayofyear
    df["is_weekend"] = (t.dayofweek >= 5).astype(int)
    us = holidays.US(years=range(df[dt].dt.year.min(), df[dt].dt.year.max() + 1))
    df["is_holiday"] = df[dt].dt.date.astype("datetime64[ns]").isin(
        pd.to_datetime(list(us.keys()))
    ).astype(int)
    if cyclical:
        # Cyclical encodings so the model sees hour 23 and hour 0 as adjacent.
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lags_rolls(df, val, lags, windows) -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = df[val].shift(lag)
    for w in windows:
        df[f"rollmean_{w}"] = df[val].shift(1).rolling(w).mean()
        df[f"rollstd_{w}"] = df[val].shift(1).rolling(w).std()
    return df


def build(df, dt, val, lags, windows, cyclical=True) -> pd.DataFrame:
    df = add_calendar(df.copy(), dt, cyclical=cyclical)
    df = add_lags_rolls(df, val, lags, windows)
    return df.dropna().reset_index(drop=True)


def main() -> None:
    p = load_params()
    df = pd.read_parquet(ROOT / "data" / "interim" / "clean.parquet")
    feats = build(
        df,
        p["data"]["datetime_col"],
        p["data"]["target_col"],
        p["features"]["lags"],
        p["features"]["rolling_windows"],
        cyclical=p["features"].get("cyclical", True),
    )
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out / "features.parquet", index=False)
    print(f"feature rows: {len(feats)}, cols: {feats.shape[1]}")


if __name__ == "__main__":
    main()
