from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


def _nan_safe_std(x: pd.Series) -> float:
    s = float(np.nanstd(x.values))
    return s if s > 1e-12 else 1.0


def preds_to_positions(
    pred: pd.Series,
    clip: float = 3.0,
    smooth_lambda: float = 0.2,
) -> pd.Series:
    """
    Map predictions to positions:
      - scale by prediction std
      - clip
      - EMA smoothing to reduce turnover
    """
    scale = _nan_safe_std(pred)
    raw = (pred / scale).clip(-clip, clip)

    pos = raw.copy()
    for t in range(1, len(pos)):
        pos.iloc[t] = (1.0 - smooth_lambda) * pos.iloc[t - 1] + smooth_lambda * raw.iloc[t]
    return pos


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


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min())


def backtest_single_asset(
    forward_returns: pd.Series,
    positions: pd.Series,
    cost_bps: float = 1.0,
) -> Perf:
    """
    forward_returns_t is realized return from holding SPX from t to t+1 (per dataset)
    positions are applied with 1-day lag to avoid lookahead: pnl_t = pos_{t-1} * ret_t
    """
    ret = forward_returns.astype(float).reset_index(drop=True)
    pos = positions.astype(float).reset_index(drop=True)

    pos_lag = pos.shift(1).fillna(0.0)
    gross_pnl = pos_lag * ret

    # transaction costs proportional to position changes
    dpos = pos.diff().abs().fillna(0.0)
    cost = (cost_bps / 1e4) * dpos  # bps to decimal
    net_pnl = gross_pnl - cost

    equity = (1.0 + net_pnl).cumprod()

    sharpe_g = _sharpe(gross_pnl)
    sharpe_n = _sharpe(net_pnl)
    mdd = _max_drawdown(equity)

    # turnover: average absolute daily position change
    turnover = float(dpos.mean())

    return Perf(
        sharpe_gross=sharpe_g,
        sharpe_net=sharpe_n,
        max_drawdown=mdd,
        turnover=turnover,
    )
