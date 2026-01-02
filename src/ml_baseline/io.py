from __future__ import annotations

from pathlib import Path

import pandas as pd


def parquet_supported() -> bool:
    try:
        import pyarrow 
    except Exception:
        return False
    return True


def read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_tabular(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        if not parquet_supported():
            raise RuntimeError(
                "Parquet requires optional dependency 'pyarrow'. Install with: uv sync --extra parquet"
            )
        df.to_parquet(path, index=False)
        return
    df.to_csv(path, index=False)


def best_effort_ext() -> str:
    return ".parquet" if parquet_supported() else ".csv"

