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
    #### Position sizing uses future information @TODO: remove future info from scaling
    ## @TODO Fix pattern: pass a scale computed from train-period predictions, or use an expanding/rolling window std with a warmup. The signature becomes preds_to_positions(pred, scale=...) — sizing state is an input, not something inferred from the series being sized.
    scale = _nan_safe_std(pred)
    raw = (pred / scale).clip(-clip, clip)

    # pos = raw.copy()
    # for t in range(1, len(pos)):
    #     pos.iloc[t] = (1.0 - smooth_lambda) * pos.iloc[t - 1] + smooth_lambda * raw.iloc[t]
    pos = raw.ewm(alpha=smooth_lambda, adjust=False).mean() # TODO adjust=True (the default!) rescales early observations
    return pos
