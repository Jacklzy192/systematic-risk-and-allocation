"""Signal-quality metrics: rank IC, block-IC significance, IC decay.

Conventions
-----------
Target alignment: `forward_returns` at row t is the return earned over
[t, t+1], so corr(feature_t, target_t) is already the 1-step-ahead IC —
no extra shift at horizon 1. Horizon h means feature_t vs the forward
return earned over [t+h-1, t+h], i.e. target.shift(-(h-1)).

Significance: the naive t-stat ic * sqrt(n) assumes n independent
observations. Persistent features (macro step functions especially) make
daily observations far from independent, inflating that t-stat badly.
Instead we compute one IC per non-overlapping block of `block_size` days
and treat block ICs as approximately iid draws:

    t = mean(block_ics) / (std(block_ics) / sqrt(n_blocks))

Block size should be >= the feature's autocorrelation half-life for the
independence assumption to be reasonable; 21 days (one month) is the
practitioner default. The block IC series also gives IC stability for
free: mean/std of block ICs is the signal's information ratio at block
frequency.

WARNING — near-integrated features (Kendall bias)
-------------------------------------------------
Block ICs are INVALID for features that random-walk within a block
(anything tracking the price level: valuation ratios, yields, vol
levels). Within-block ranks are computed against the whole block —
including future days — and for an integrated series this manufactures
spurious mean reversion: under a null of iid returns, a random-walk
level shows block IC ~= -0.31 at block_size=21 (verified by simulation;
the classic short-window AR bias, ~ -(1+3*rho)/T). Symptom: |block IC|
huge while full-sample IC ~= 0, and ic_fwd ~= -ic_bwd.

Fix: run ic_table with transform="diff" for persistent features, and
use the ac1 column to spot them (ac1 > ~0.95 means the level-IC rows
are artifact). The right long-term answer is transforming features to
stationarity (causal z-scores / changes) before any IC analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MIN_PAIRS_FULL = 100  # minimum valid pairs for a full-sample IC
MIN_PAIRS_BLOCK = 10  # minimum valid pairs within one block


def rank_ic(feature: pd.Series, target: pd.Series) -> float:
    """Spearman rank IC on pairwise-complete observations."""
    mask = feature.notna() & target.notna()
    if mask.sum() < MIN_PAIRS_FULL:
        return np.nan
    f, y = feature[mask], target[mask]
    if f.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(spearmanr(f, y)[0])


def block_ics(
    feature: pd.Series,
    target: pd.Series,
    block_size: int = 21,
) -> np.ndarray:
    """
    Rank IC per non-overlapping block of `block_size` consecutive rows.

    Blocks are cut on the raw (calendar) index, THEN NaNs are dropped
    within each block — so a block never spans a long missing stretch
    pretending to be contiguous time. Blocks with too few valid pairs or
    a constant feature (e.g. an inactive dummy) are skipped.
    """
    n_blocks = len(feature) // block_size
    ics: List[float] = []
    for b in range(n_blocks):
        sl = slice(b * block_size, (b + 1) * block_size)
        f, y = feature.iloc[sl], target.iloc[sl]
        mask = f.notna() & y.notna()
        if mask.sum() < MIN_PAIRS_BLOCK:
            continue
        f, y = f[mask], y[mask]
        if f.nunique() < 2 or y.nunique() < 2:
            continue
        ic = spearmanr(f, y)[0]
        if np.isfinite(ic):
            ics.append(float(ic))
    return np.asarray(ics)


def ic_table(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    block_size: int = 21,
    transform: str | None = None,
) -> pd.DataFrame:
    """
    One row per feature, sorted by |block t-stat|:

      ic_full       full-sample rank IC (point estimate; do NOT use its
                    naive t-stat — see module docstring)
      ic_mean       mean of block ICs
      ic_std        std of block ICs (IC volatility — stability measure)
      ic_ir         ic_mean / ic_std (per-block information ratio)
      t_stat        block-based t-stat, the significance number to trust
      ac1           lag-1 autocorrelation of the (raw) feature; > ~0.95
                    means the level-based block IC is untrustworthy
                    (Kendall bias — see module docstring)
      n_blocks      blocks that produced a valid IC
      n_obs         valid (feature, target) pairs in the full sample

    transform="diff" first-differences each feature (causal: uses only
    t-1 and t) — the required mode for persistent features.

    With ~94 features, expect ~5 to clear |t| > 2 by chance alone —
    read the table with a multiple-testing discount (|t| > 3 is a more
    honest bar for "this feature has signal").
    """
    if transform not in (None, "diff"):
        raise ValueError(f"transform must be None or 'diff', got {transform!r}")
    target = df[target_col]
    rows = []
    for col in feature_cols:
        raw = df[col]
        feat = raw.diff() if transform == "diff" else raw
        ics = block_ics(feat, target, block_size=block_size)
        n_blocks = len(ics)
        if n_blocks >= 2:
            ic_mean = float(ics.mean())
            ic_std = float(ics.std(ddof=1))
            se = ic_std / np.sqrt(n_blocks)
            t_stat = ic_mean / se if se > 1e-12 else np.nan
            ic_ir = ic_mean / ic_std if ic_std > 1e-12 else np.nan
        else:
            ic_mean = ic_std = t_stat = ic_ir = np.nan
        rows.append(
            {
                "feature": col,
                "ic_full": rank_ic(feat, target),
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "t_stat": t_stat,
                "ac1": float(raw.autocorr(1)) if raw.nunique() > 1 else np.nan,
                "n_blocks": n_blocks,
                "n_obs": int((feat.notna() & target.notna()).sum()),
            }
        )
    out = pd.DataFrame(rows).set_index("feature")
    return out.reindex(out["t_stat"].abs().sort_values(ascending=False).index)


def ic_decay(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    horizons: Iterable[int] = (1, 2, 5, 10, 21),
) -> pd.DataFrame:
    """
    Rank IC of feature_t vs the forward return h steps ahead, per horizon.

    Rows = features, columns = horizons. A signal worth trading at daily
    frequency should peak at h=1 and decay smoothly; an IC that RISES with
    horizon on a slow macro feature usually means the feature is tracking
    a regime, not predicting tomorrow — model it accordingly (or suspect
    alignment problems in the data).
    """
    target = df[target_col]
    out = pd.DataFrame(index=list(feature_cols), columns=list(horizons), dtype=float)
    for h in horizons:
        shifted = target.shift(-(h - 1))
        for col in out.index:
            out.loc[col, h] = rank_ic(df[col], shifted)
    out.index.name = "feature"
    out.columns.name = "horizon"
    return out


@dataclass(frozen=True)
class Perf:
    sharpe_gross: float
    sharpe_net: float
    max_drawdown: float
    turnover: float


def _sharpe(daily_pnl: pd.Series) -> float:
    mu = daily_pnl.mean()
    sd = daily_pnl.std()
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(252.0))

def _max_drawdown(daily_pnl: pd.Series) -> float:
    """
    Additive max drawdown in return units: min_t (cum_pnl_t - peak_of_cum_pnl).

    Takes DAILY pnl, not an equity curve. Additive (simple cumulative pnl)
    rather than compounded because a compounded curve misbehaves when
    leverage pushes equity near/below zero — (equity - peak)/peak flips
    sign past zero and drawdowns exceed 100%. The additive frame is the
    standard one for signal evaluation; compounding is a portfolio
    construction question. Returns a negative number (or 0.0).
    """
    cum = daily_pnl.cumsum()
    return float((cum - cum.cummax()).min())