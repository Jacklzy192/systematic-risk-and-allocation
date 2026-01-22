````markdown
# systematic-risk-and-allocation

Systematic portfolio risk analysis and signal-to-allocation research using macro and market data.

---

## Project: Systematic Portfolio Risk & Allocation

### Objective
Build a systematic framework to translate return forecasts into position sizing under risk and volatility constraints.

---

## Data
- Daily macro & market features (Kaggle-style dataset)
- Feature groups include:
  - `M*` Market Dynamics/Technical
  - `E*` Macro Economic
  - `I*` Interest Rate
  - `P*` Price/Valuation
  - `V*` Volatility
  - `S*` Sentiment
  - `MOM*` Momentum
  - `D*` Dummy/Binary features
- Target:
  - `forward_returns`: next-day S&P 500 forward return (train only)

**Notes**
- The public leaderboard test set is not meaningful by design. Model evaluation is performed via time-based splits on the training history.

---

## Models
- LightGBM baseline (primary)
- Optional extensions: CatBoost / XGBoost ensemble (future)

Focus: stability and out-of-sample performance (time-series evaluation), not in-sample fitting.

---

## Portfolio Construction
- Prediction → signal mapping with clipping and scaling
- Position sizing with turnover-aware smoothing (baseline)
- Emphasis on risk-adjusted performance, not raw prediction error

---

## Design Philosophy
This project is structured to mirror a professional systematic strategy pipeline:

**Signal Research → Risk Assessment → Portfolio Construction**

The emphasis is not on maximizing in-sample accuracy, but on ensuring that each layer remains robust across different market regimes (e.g., high volatility, liquidity stress). Feature engineering incorporates proxies for macro-financial conditions, informed by balance-sheet and funding considerations, to better capture regime shifts and changing risk dynamics.

---

## How to Run

### 1) Place data files locally (not committed)
Download the Kaggle dataset and place:
- `train.csv`
- `test.csv`

into: `data/raw/`

### 2) Create environment + install dependencies
```bash
conda create -n sra python=3.11 -y
conda activate sra
pip install -r requirements.txt
```

### 3) Run baseline pipeline
```bash
PYTHONPATH=. python scripts/run_baseline.py
```

This will:
- Train a LightGBM baseline using a time-based holdout split
- Map predictions to positions
- Backtest with transaction costs
- Save predictions to: `artifacts/valid_preds.csv`

---

## Results (current baseline)

| Model         | Sharpe (gross) | Sharpe (net, 1bp cost) | Max DD (net) | Turnover |
|:--------------|:--------------:|:----------------------:|:------------:|:--------:|
| Baseline LGBM | 0.864          | 0.855                  | -54.9%       | 0.112    |

### Key Takeaways
- Model accuracy alone is insufficient; allocation and risk controls dominate performance
- Regime shifts materially affect risk behavior
- Position mapping is the most important and most fragile layer
- Evaluation focuses on volatility-adjusted Sharpe, drawdown, turnover, and cost-adjusted returns rather than RMSE alone

---

## Status / Roadmap

✅ Baseline pipeline implemented (ingestion → time split → model → signal → backtest)

🔜 **Next steps:**
- Walk-forward validation (multiple rolling folds)
- Feature engineering v1 (rolling z-scores, volatility regimes, PCA factors)
- Vol targeting + leverage caps for improved drawdown control
- More realistic cost/slippage modeling
- Regime-sliced performance diagnostics
````

---
