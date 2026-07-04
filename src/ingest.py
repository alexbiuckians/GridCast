"""Stage 1: ingest raw CSV -> parsed, sorted parquet."""
import pandas as pd

from src.utils import ROOT, load_params


def ingest(raw_path: str, datetime_col: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    return df


def main() -> None:
    p = load_params()
    out = ROOT / "data" / "interim"
    out.mkdir(parents=True, exist_ok=True)
    df = ingest(p["data"]["raw_path"], p["data"]["datetime_col"])
    df.to_parquet(out / "ingested.parquet", index=False)
    print(f"ingested {len(df)} rows")


if __name__ == "__main__":
    main()
