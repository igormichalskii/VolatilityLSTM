import yfinance as yf
import numpy as np
import pandas as pd

def fetch_and_clean_data(symbol: str="SPY", vix_symbol: str="^VIX", start_date: str="2010-01-01"):
    print(f"1. Fetching data for {symbol} and {vix_symbol}...")

    # Extract the main asset
    spy = yf.Ticker(symbol).history(start=start_date)[['Close']]
    spy.rename(columns={'Close': 'close'}, inplace=True)
    # Strip timezones and set the time to exactly midnight.
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    
    # Extract the Fear Gauge
    vix = yf.Ticker(vix_symbol).history(start=start_date)[['Close']]
    vix.rename(columns={'Close': 'vix_close'}, inplace=True)
    # Strip timezones and set the time to exactly midnight.
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()

    print("2. Merging datasets and calculating targets...")
    # Combine them. The 'inner' join ensures we only keep days where BOTH markets were open.
    df = spy.join(vix, how='inner')

    # The Math
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    window = 21
    annualization_factor = np.sqrt(252)
    df['realized_vol'] = df['log_return'].rolling(window=window).std() * annualization_factor

    # The Purge
    clean_df = df.dropna()
    output_file = f"{symbol}_VIX_daily_clean.parquet"
    clean_df.to_parquet(output_file)

    print(f"3. Data scrubbed and saved to {output_file}. Shape: {clean_df.shape}")
    return clean_df

if __name__ == "__main__":
    fetch_and_clean_data()