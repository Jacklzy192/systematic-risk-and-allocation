from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from src.signal import preds_to_positions
from src.metrics import _sharpe, _max_drawdown, Perf

"""
TODO: Add tests for backtest_single_asset.
9. No tests yet
For this codebase the highest-value tests are cheap and quant-specific:

Leakage test: assert load_train(...).feature_cols ∩ LEAK_COLS_TRAIN is empty, and that train/test feature columns are identical.
Convention test: feed backtest_single_asset a synthetic series where you know the answer (e.g., perfect foresight positions → Sharpe should be huge; zero positions → exactly 0 pnl, 0 cost). This pins the lag convention so #2 can never silently regress.
Split test: assert max(train_idx) + gap < min(valid_idx).
These are regression insurance for the invariants, which is what testing looks like in research code (vs. testing exact numbers, which is brittle).
"""

def backtest_single_asset(
    forward_returns: pd.Series,
    positions: pd.Series,
    execution_lag: int = 0,
    cost_bps: float = 1.0,
) -> Perf:
    """
    forward_returns_t is realized return from holding SPX from t to t+1 (per dataset)
    The execution lag: an extra delay is a legitimate execution-delay stress test 
    cost_bps: transaction costs in basis points (bps)
    """
    ret = forward_returns.astype(float).reset_index(drop=True)
    pos = positions.astype(float).reset_index(drop=True)

    pos_lag = pos.shift(execution_lag).fillna(0.0)
    gross_pnl = pos_lag * ret

    # transaction costs proportional to position changes
    dpos = pos.shift(execution_lag).diff().abs().fillna(0.0) ### TODO check if the pos_lag and dpos are aligned correctly
    cost = (cost_bps / 1e4) * dpos  # bps to decimal
    net_pnl = gross_pnl - cost

    sharpe_g = _sharpe(gross_pnl)
    sharpe_n = _sharpe(net_pnl)
    mdd = _max_drawdown(net_pnl)  # takes daily pnl (additive frame), not equity

    # turnover: average absolute daily position change
    turnover = float(dpos.mean())

    return Perf(
        sharpe_gross=sharpe_g,
        sharpe_net=sharpe_n,
        max_drawdown=mdd,
        turnover=turnover,
    )
