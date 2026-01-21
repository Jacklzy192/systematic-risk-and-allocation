from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


LEAK_COLS_TRAIN = {
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
}

NON_FEATURE_COLS_COMMON = {"date_id", "is_scored"}  # date_id is index/meta, not a feature


@dataclass(frozen=True)
class Dataset:
    df: pd.DataFrame
    feature_cols: List[str]
    target_col: str = "forward_returns"
    time_col: str = "date_id"


def _infer_feature_cols(df: pd.DataFrame, target_col: str) -> List[str]:
    cols = list(df.columns)
    drop = set(NON_FEATURE_COLS_COMMON)
    drop.add(target_col)

    # drop any obvious target/leak columns if present
    drop |= LEAK_COLS_TRAIN

    # drop lagged labels from test-like frames if user accidentally points to it
    for c in ["lagged_forward_returns", "lagged_risk_free_rate", "lagged_market_forward_excess_returns"]:
        if c in df.columns:
            drop.add(c)

    feature_cols = [c for c in cols if c not in drop]
    return feature_cols


def load_train(data_dir: str | Path) -> Dataset:
    data_dir = Path(data_dir)
    path = data_dir / "train.csv"
    df = pd.read_csv(path)

    # basic checks
    if "date_id" not in df.columns:
        raise ValueError("train.csv must contain 'date_id'")
    if "forward_returns" not in df.columns:
        raise ValueError("train.csv must contain target column 'forward_returns'")

    df = df.sort_values("date_id").reset_index(drop=True)

    feature_cols = _infer_feature_cols(df, target_col="forward_returns")

    # keep only relevant columns to avoid accidental leakage downstream
    keep = ["date_id", "forward_returns"] + feature_cols
    df = df[keep]

    return Dataset(df=df, feature_cols=feature_cols)


def load_test(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    path = data_dir / "test.csv"
    df = pd.read_csv(path)

    if "date_id" not in df.columns:
        raise ValueError("test.csv must contain 'date_id'")

    df = df.sort_values("date_id").reset_index(drop=True)
    return df
