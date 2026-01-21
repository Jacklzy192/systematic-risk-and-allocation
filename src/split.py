from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_idx: np.ndarray
    valid_idx: np.ndarray


def simple_time_split(df: pd.DataFrame, valid_frac: float = 0.15) -> TimeSplit:
    """
    Single holdout split by time. No shuffling.
    """
    n = len(df)
    if not (0.05 <= valid_frac <= 0.5):
        raise ValueError("valid_frac should be in [0.05, 0.5]")

    split = int(n * (1.0 - valid_frac))
    train_idx = np.arange(0, split)
    valid_idx = np.arange(split, n)
    return TimeSplit(train_idx=train_idx, valid_idx=valid_idx)
