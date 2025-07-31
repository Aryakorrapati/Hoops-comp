"""
Load CSVs once, cache the DataFrames, and expose helpers.
Put *all* your CSV/TSV files inside project/data.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

_cache: dict[str, pd.DataFrame] = {}

def load_csv(name: str) -> pd.DataFrame:
    if name not in _cache:
        _cache[name] = pd.read_csv(DATA_DIR / name)
    return _cache[name]

def pace_df() -> pd.DataFrame:
    return load_csv("pace.csv")

def cbb_df() -> pd.DataFrame:
    return load_csv("cbb_stats.csv")

def nba_df() -> pd.DataFrame:
    return load_csv("nba_stats.csv")
