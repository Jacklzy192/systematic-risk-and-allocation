# systematic-risk-and-allocation
Systematic portfolio risk analysis and signal-to-allocation research using macro and market data

## Project: Systematic Portfolio Risk & Allocation

### Objective
Build a systematic framework to translate return forecasts into position sizing under risk and volatility constraints.

### Data
- Daily macro & market features
- Engineered features: rolling z-scores, volatility regimes, PCA factors
- Forward returns as prediction target

### Models
- LightGBM / CatBoost / XGBoost ensemble
- Focus on stability and out-of-sample performance

### Portfolio Construction
- Signal-to-position mapping with volatility and drawdown awareness
- Emphasis on risk-adjusted performance, not raw prediction error

### Design Philosophy

This project is structured to mirror a professional systematic strategy pipeline: 
Signal Research → Risk Assessment → Portfolio Construction.

The emphasis is not on maximizing in-sample accuracy, but on ensuring that each layer remains robust across different market regimes (e.g., high volatility, liquidity stress). Feature engineering incorporates proxies for macro-financial conditions, informed by balance-sheet and funding considerations, to better capture regime shifts and changing risk dynamics.


### Key Takeaways
- Model accuracy alone is insufficient; allocation and risk controls dominate performance
- Regime shifts materially affect risk behavior
- Position mapping is the most important and most fragile layer
- Evaluation focused on volatility-adjusted Sharpe and drawdown behavior rather than RMSE

## Status
Ongoing research; code and analysis to be added incrementally.
