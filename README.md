# Volatility Showdown: GARCH(1,1) vs. Fear-Injected LSTM

An end-to-end quantitative research pipeline comparing classical econometrics against deep learning architecture for forecasting the 21-day rolling realized volatility of the S&P 500 (SPY).

This project demonstrates how integrating exogenous market sentiment data (the VIX index) empowers an LSTM's recursive memory architecture to dramatically outperform standard autoregressive statistical models.

## The Problem: GARCH Overreaction vs. Regime Shifts

Financial asset volatility is non-stationary and moves in regimes. Standard GARCH(1,1) models excel in mean-reverting environments but catastrophically overreact to singular market shocks (black swan days). GARCH models lack context; they only know history.

## The Solution: A Fear-Injected LSTM

This architecture exploits the **LSTM's Forget Gate** technology by combining autoregressive history (past S&P 500 returns) with immediate market sentiment (raw VIX closing prices). By ingesting the VIX, the LSTM gains "contextual memory," allowing it to distinguish between a singular market crash day (noise) and a sustained high-volatility regime (signal).

## The Final Performance Metric

The pipeline was executed on out-of-sample data, testing the models against realities they had never seen. The evaluation Refereed via Root Mean Squared Error (RMSE), heavily penalizing massive prediction failures.

| Model | Architecture | Root Mean Squared Error (RMSE) |
| :--- | :--- | :--- |
| **GARCH(1,1)** | Parameter-Fixed Baseline (`arch`) | **0.0413** |
| **LSTM (Optimized)** | **2-Layer Deep Stack + VIX Input** | **0.0232 (Winner)** |

The optimized LSTM reduced volatility forecast error by **~43.8%** compared to the classical baseline.

### Key Observation (Visual Analysis)

The resulting forecast comparison clearly isolates why the LSTM won: during the 2025 market crash (refer to the chart generated in the notebook), GARCH overreacted violently, overshooting reality by nearly 100%. The LSTM, stabilized by the VIX data, filtered the singular day's panic and tracked the true realized volatility curve with remarkable precision.

[Link to final chart graphic placeholder]

## Project Reproducibility (Step-by-Step)

### Prerequisites (Python Environment)

Ensure your environment has PyTorch (for the LSTM) and the econometric libraries installed:
```bash
pip install yfinance pandas numpy torch arch scikit-learn matplotlib pyarrow notebook