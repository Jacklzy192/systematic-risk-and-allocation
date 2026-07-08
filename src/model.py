from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor




@dataclass
class LinearModel:
    model: Ridge | Lasso
    feature_cols: list[str]
    model_type: str

@dataclass
class XGBoostModel:
    model: XGBRegressor
    feature_cols: list[str]
    model_type: str = "xgboost"

@dataclass
class CatBoostModel:
    model: CatBoostRegressor
    feature_cols: list[str]
    model_type: str = "catboost"

@dataclass
class LGBMModel:
    model: LGBMRegressor
    feature_cols: list[str]
    model_type: str = 'lgbm'

def train_linear(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'ridge',
    alpha: float = 1.0
) -> LinearModel:
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha)
    else:
        raise ValueError(f"unknown model type: {model_type}")
    model.fit(X_train, y_train)
    return LinearModel(model, feature_cols=list(X_train.columns), model_type=model_type)

def train_xgboost(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        params: Dict | None = None,
) -> XGBoostModel:
    default_params =  dict(
        n_estimators=2000,   # more boosting rounds
        max_depth=7,         # deeper trees
        learning_rate=0.01,  # smaller LR for stability
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=1,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=200,
    )
    if params:
        default_params.update(params)
    model = XGBRegressor(**default_params)
    model.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)], 
              verbose=False)
    return XGBoostModel(model=model, feature_cols=list(X_train.columns))

def train_catboost(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        params: Dict | None = None,
) -> CatBoostModel:
    
    default_params = dict(
        iterations=2000,
        depth=7,
        learning_rate=0.01,
        l2_leaf_reg=3,
        random_seed=42,
        loss_function='RMSE',
        early_stopping_rounds=200,
        verbose=False
    )
    if params:
        default_params.update(params)
    model = CatBoostRegressor(**default_params)
    model.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)], 
              verbose=False)
    return CatBoostModel(model=model, feature_cols=list(X_train.columns))
    
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

    model = LGBMRegressor(**default_params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
    
    return LGBMModel(model=model, feature_cols=list(X_train.columns))


def predict(model, X: pd.DataFrame) -> np.ndarray:
    X = X[model.feature_cols]
    return model.model.predict(X)