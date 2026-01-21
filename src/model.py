from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb


@dataclass
class LGBMModel:
    booster: lgb.Booster
    feature_cols: list[str]


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Dict | None = None,
) -> LGBMModel:
    default_params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=64,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
        seed=42,
    )
    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        default_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )
    return LGBMModel(booster=booster, feature_cols=list(X_train.columns))


def predict(model: LGBMModel, X: pd.DataFrame) -> np.ndarray:
    X = X[model.feature_cols]
    return model.booster.predict(X, num_iteration=model.booster.best_iteration)
