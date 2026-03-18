# Volatility Showdown: Econometrics vs. Optuna-Optimized Deep Learning

An end-to-end quantitative research pipeline comparing classical econometrics against a hyperparameter-optimized deep learning architecture for forecasting the 21-day rolling realized volatility of the S&P 500 (SPY).

This project proves that integrating exogenous market sentiment data (the VIX index) with a Bayesian-optimized LSTM architecture dramatically outperforms standard autoregressive statistical models in out-of-sample chaotic markets.

## The Problem: GARCH Overreaction vs. Regime Shifts

Financial asset volatility is non-stationary and moves in regimes. Standard GARCH(1,1) models excel in mean-reverting environments but catastrophically overreact to singular market shocks. GARCH models lack macroeconomic context; they only know isolated price history.

## The Solution: A Fear-Injected, Bayesian-Tuned LSTM

This architecture exploits the **LSTM's Forget Gate** technology by combining autoregressive history (past S&P 500 returns) with immediate market sentiment (raw VIX closing prices). 

Instead of manually guessing the network architecture, the model was subjected to rigorous Bayesian optimization via **Optuna**. The Thunderdome proved that a wide, shallow network (128 hidden dimensions, 1 layer) optimally ingested the fear index without memorizing the noise, aggressively beating deeper, narrower networks.

## The Final Performance Metric

The pipeline was executed on out-of-sample data, testing the models against realities they had never seen. The evaluation Refereed via Root Mean Squared Error (RMSE), heavily penalizing massive prediction failures.

| Model | Architecture | Root Mean Squared Error (RMSE) |
| :--- | :--- | :--- |
| **GARCH(1,1)** | Parameter-Fixed Baseline (`arch`) | 0.0413 |
| **LSTM (Manual)** | 64 Lanes, 2 Layers | 0.0232 |
| **LSTM (Optuna Champion)** | **128 Lanes, 1 Layer** | **0.0208 (Winner)** |

The optimized LSTM reduced volatility forecast error by **~49.6%** compared to the classical baseline.

### Key Observation (Visual Analysis)

During the mid-2025 market crash, GARCH overreacted violently, overshooting reality by nearly 100%. The Optuna-optimized LSTM, stabilized by the VIX data and a wide memory highway, filtered the singular day's panic and tracked the true realized volatility curve with remarkable precision.

[Link to final chart graphic placeholder]

## Project Reproducibility (Step-by-Step)

### Prerequisites
Ensure your environment has PyTorch and the econometric/optimization libraries installed:
`pip install yfinance pandas numpy torch arch scikit-learn matplotlib pyarrow notebook optuna`

### 1. Data Ingestion & Sanitization (`data_pipeline.py`)
Pulls daily `SPY` and `^VIX` history via `yfinance`. Sanitizes dates to strip timezones (preventing empty join arrays), merges the datasets, and engineers the 21-day rolling realized volatility target. Saved to Parquet for optimized I/O.
*Run: `python data_pipeline.py`*

### 2. The Gladiator Pit (`train.py`)
Executes the main loop. Separates data via a strict chronological 80/20 train/test split. Trains the GARCH baseline, prepares the input sequences, and unleashes an Optuna study. Optuna breeds and evaluates multiple LSTM architectures, locks the random seed for reproducibility, and trains the final champion on the optimal parameters.

### 3. Presentation & Descaling (`showdown.ipynb`)
The primary interface. Executes the optimized pipeline, descales the final LSTM outputs back into annualized volatility percentages, perfectly aligns the asynchronous arrays, calculates the final RMSE scores, and renders the Matplotlib comparison chart.