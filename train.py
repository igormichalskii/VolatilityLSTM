import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model
from volatility_models import VolatilityLSTM

def create_sequences(data, seq_length): # Helper function to create sequences of data for LSTM training
    xs, ys = [], [] 
    for i in range(len(data) - seq_length): # Loop through the data to create sequences of the specified length
        xs.append(data[i:(i+seq_length)])
        # Index 1 is still 'realized_vol' based on our features list below
        ys.append(data[i+seq_length,1])
    return np.array(xs), np.array(ys)

def run_training_pipeline(file_path="SPY_VIX_daily_clean.parquet"): # Main function to run the training pipeline
    df = pd.read_parquet(file_path) 
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # --- GARCH Walk-Forward ---
    am_train = arch_model(train_df['log_return'] * 100, vol='Garch', p=1, q=1, rescale=False) # Fit the GARCH(1,1) model to the training data (log returns scaled by 100 for percentage)
    train_res = am_train.fit(disp='off') # Fit the model without printing output to the console

    am_full = arch_model(df['log_return'] * 100, vol='Garch', p=1, q=1, rescale=False) # Fit the GARCH(1,1) model to the full dataset to get the conditional volatility for the test period
    full_res = am_full.fix(train_res.params) # Use the parameters from the training fit to ensure consistency in the volatility estimates

    garch_daily_vol = full_res.conditional_volatility / 100 # Convert back to daily volatility by dividing by 100 (since we scaled log returns by 100)
    garch_annual_vol = garch_daily_vol * np.sqrt(252) # Annualize the daily volatility by multiplying by the square root of the number of trading days in a year
    garch_predictions = garch_annual_vol.values[split_idx:] # Extract the GARCH predictions for the test period to compare against the LSTM predictions

    # --- LSTM Training ---
    print("Prepping and Training LSTM...")
    scaler = MinMaxScaler()

    # THE UPGRADE: We added the VIX to the feature list
    features = ['log_return', 'realized_vol', 'vix_close']
    train_scaled = scaler.fit_transform(train_df[features]) # Scale the features for the training data using MinMaxScaler to improve LSTM training stability
    test_scaled = scaler.transform(test_df[features])

    seq_length = 21
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # Convert training data to PyTorch tensors and add an extra dimension to the target variable for compatibility with the LSTM output

    # Instantiate the upgraded model (it defaults to 3 inputs now)
    model = VolatilityLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    epochs = 50
    for epoch in range(epochs): # Train the LSTM model for the specified number of epochs, performing a forward pass, calculating the loss, and updating the model parameters using backpropagation
        model.train() 
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
    
    print("Pipeline Ready.")
    return test_df, garch_predictions, model, X_test, scaler, seq_length

if __name__ == "__main__":
    run_training_pipeline()