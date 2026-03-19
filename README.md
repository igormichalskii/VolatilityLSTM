# VolatilityLSTM: Institutional-Grade Market Forecasting

This repository contains a brutally optimized Deep Learning pipeline designed to forecast 21-day realized volatility for the S&P 500. It abandons academic static testing in favor of ruthless, real-world quant infrastructure.

### The Engine
* **The Architecture:** PyTorch LSTM dynamically sized via Bayesian search (`Optuna`).
* **The Lens:** `StandardScaler` to feed the network pure statistical anomalies (Z-scores) rather than crushed absolute values.
* **The Psychology:** Custom `AsymmetricVolatilityLoss` function. The model is penalized 3x harder for under-predicting volatility, forcing it to respect black swan events and tail-risk.
* **The Time Machine:** Purged Rolling Walk-Forward Validation. The model maintains a strict memory buffer, continuously inducing selective amnesia and retraining itself across multiple simulated eras to eliminate look-ahead bias and adapt to shifting macroeconomic regimes.

### The Scoreboard

| The Tech Stack | RMSE | The Reality |
| :--- | :--- | :--- |
| GARCH(1,1) Baseline | 0.0413 | The 1980s academic baseline. Panics easily. |
| Static LSTM (MinMax Scaler) | 0.0208 | Standard look-ahead bias illusion. |
| **Walk-Forward LSTM (The Apex)** | **0.0132** | **Institutional-grade alpha.** |

### Execution
1. Mine the pristine data: `python data_pipeline.py`
2. Enter the Thunderdome: Run `showdown.ipynb` to execute the rolling walk-forward simulation, induce the amnesia loop, and generate the final out-of-sample RMSE.