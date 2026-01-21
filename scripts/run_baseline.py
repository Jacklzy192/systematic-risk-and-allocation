from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion import load_train
from src.split import simple_time_split
from src.model import train_lgbm, predict
from src.backtest import preds_to_positions, backtest_single_asset


DATA_DIR = Path("data/raw")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ds = load_train(DATA_DIR)
    df = ds.df.copy()

    # Basic missing handling: LightGBM can handle NaNs.
    X = df[ds.feature_cols]
    y = df[ds.target_col]

    split = simple_time_split(df, valid_frac=0.15)
    X_tr, y_tr = X.iloc[split.train_idx], y.iloc[split.train_idx]
    X_va, y_va = X.iloc[split.valid_idx], y.iloc[split.valid_idx]

    model = train_lgbm(X_tr, y_tr, X_va, y_va)

    pred_va = predict(model, X_va)
    pred_va = pd.Series(pred_va, index=y_va.index, name="pred")

    pos_va = preds_to_positions(pred_va, clip=3.0, smooth_lambda=0.2)

    perf = backtest_single_asset(
        forward_returns=y_va.reset_index(drop=True),
        positions=pos_va.reset_index(drop=True),
        cost_bps=1.0,
    )

    out = pd.DataFrame(
        {
            "date_id": df.loc[split.valid_idx, ds.time_col].values,
            "y": y_va.values,
            "pred": pred_va.values,
            "pos": pos_va.values,
        }
    )
    out_path = ART_DIR / "valid_preds.csv"
    out.to_csv(out_path, index=False)

    print("\n=== Baseline (Validation) ===")
    print(f"Rows: {len(out):,}")
    print(f"Sharpe (gross): {perf.sharpe_gross:.3f}")
    print(f"Sharpe (net, 1bp cost): {perf.sharpe_net:.3f}")
    print(f"Max Drawdown (net): {perf.max_drawdown:.3%}")
    print(f"Turnover (avg |Δpos|): {perf.turnover:.4f}")
    print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    main()
