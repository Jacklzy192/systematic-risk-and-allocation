from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_idx: np.ndarray
    valid_idx: np.ndarray

"""
This is López de Prado's purging, in its simplest form. 
Also: your commit message says "walk-forward split" but this is a single holdout — 
for interview-facing work, either TODO rename it or actually implement expanding-window walk-forward 
(which you'll want anyway; a single 15% holdout on one regime is a weak estimate of out-of-sample Sharpe, 
and you know from the stat-arb side how regime-dependent SPX timing signals are)."""
### TODO with this gap, is some day got purged twice? (once in train, once in valid) — if so, is that a problem? 
### np.arange is exclude the end, so I think it's fine. But check with a small example to be sure.
def simple_time_split(df: pd.DataFrame, valid_frac: float = 0.15, gap=1) -> TimeSplit:
    """
    Single holdout split by time. No shuffling.
    """
    n = len(df)
    if not (0.05 <= valid_frac <= 0.5):
        raise ValueError("valid_frac should be in [0.05, 0.5]")
    split = int(n * (1.0 - valid_frac))
    if split - gap <= 0:
        raise ValueError(f"gap ({gap}) too large for train size ({split})")
    
    train_idx = np.arange(0, split-gap)
    valid_idx = np.arange(split, n)
    assert max(train_idx) + gap < min(valid_idx), f"train/valid gap violated: {max(train_idx)} + {gap} >= {min(valid_idx)}"
    return TimeSplit(train_idx=train_idx, valid_idx=valid_idx)


@dataclass(frozen=True)
class LockboxSplit:
    research_idx: np.ndarray
    lockbox_idx: np.ndarray


def lockbox_split(df: pd.DataFrame, lockbox_days: int = 1000) -> LockboxSplit:
    """
    Reserve the final `lockbox_days` rows as a lockbox, untouched by EDA,
    feature selection, and model iteration. Opened ONCE, at the very end,
    to estimate true out-of-sample performance.

    Why: any full-sample statistic (IC rankings, PCA, regime segmentation)
    that influences a modeling decision contaminates evaluation on that same
    period — data snooping. Everything downstream (EDA notebooks,
    simple_time_split, walk-forward) should operate on research_idx only.

    Indices are positional: consume with .iloc, never .loc.
    """
    n = len(df)
    if not (0 < lockbox_days < n):
        raise ValueError(f"lockbox_days must be in (0, {n}), got {lockbox_days}")

    split = n - lockbox_days
    return LockboxSplit(
        research_idx=np.arange(0, split),
        lockbox_idx=np.arange(split, n),
    )
