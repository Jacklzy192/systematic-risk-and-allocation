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
class ResearchDataset:
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

    # # drop lagged labels from test-like frames if user accidentally points to it
    # for c in ["lagged_forward_returns", "lagged_risk_free_rate", "lagged_market_forward_excess_returns"]:
    #     if c in df.columns:
    #         drop.add(c)

    feature_cols = [c for c in cols if c not in drop]
    return feature_cols


def load_data(
    data_dir: str | Path,
    file_name: str = "train.csv",
    feature_cols: List[str] | None = None,
) -> ResearchDataset:
    data_dir = Path(data_dir)
    path = data_dir / file_name
    df = pd.read_csv(path)

    # basic checks
    if "date_id" not in df.columns:
        raise ValueError(f"{file_name} must contain 'date_id'")

    df = df.sort_values("date_id").reset_index(drop=True)

    has_target = "forward_returns" in df.columns

    # infer feature columns from this frame, or reuse train's columns (e.g. for test)
    if feature_cols is None:
        feature_cols = _infer_feature_cols(df, target_col="forward_returns")

    # keep only relevant columns to avoid accidental leakage downstream
    keep = ["date_id"] + (["forward_returns"] if has_target else []) + feature_cols
    df = df[keep]

    return ResearchDataset(df=df, feature_cols=feature_cols)


def load_train(data_dir: str | Path, file_name: str = "train.csv") -> ResearchDataset:
    return load_data(data_dir, file_name)


def load_test(
    data_dir: str | Path,
    feature_cols: List[str],
    file_name: str = "test.csv",
) -> ResearchDataset:
    return load_data(data_dir, file_name, feature_cols=feature_cols)
