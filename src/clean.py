"""Stage 2: clean -> dedupe DST timestamps, enforce hourly grid, bound values."""
import pandas as pd

from src.utils import ROOT, load_params


def clean(df: pd.DataFrame, dt: str, val: str, vmin: float, vmax: float) -> pd.DataFrame:
    # Drop duplicate timestamps (DST fall-back artifacts), keep first.
    df = df.drop_duplicates(subset=dt, keep="first").copy()
    # Reindex onto a continuous hourly grid; interpolate small gaps.
    df = df.set_index(dt).sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full)
    df[val] = df[val].interpolate(method="time", limit=6)
    df = df.dropna(subset=[val])
    # Sanity bounds.
    df = df[(df[val] >= vmin) & (df[val] <= vmax)]
    df.index.name = dt
    return df.reset_index()


def main() -> None:
    p = load_params()
    src = ROOT / "data" / "interim" / "ingested.parquet"
    df = pd.read_parquet(src)
    out = clean(
        df,
        p["data"]["datetime_col"],
        p["data"]["target_col"],
        p["data"]["value_min"],
        p["data"]["value_max"],
    )
    out.to_parquet(ROOT / "data" / "interim" / "clean.parquet", index=False)
    print(f"clean rows: {len(out)}")


if __name__ == "__main__":
    main()
