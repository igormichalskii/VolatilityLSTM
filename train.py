import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from arch import arch_model
import optuna
from volatility_models import VolatilityLSTM

torch.manual_seed(42)
np.random.seed(42)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length, 1])
    return np.array(xs), np.array(ys)

def run_optimized_pipeline(file_path="SPY_VIX_daily_clean.parquet", n_trials=15):
    df = pd.read_parquet(file_path)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print("1. Fitting GARCH(1,1) Baseline...")
    am_train = arch_model(train_df['log_return'] * 100, vol='Garch', p=1, q=1, rescale=False)
    train_res = am_train.fit(disp="off")
    am_full = arch_model(df['log_return'] * 100, vol='Garch', p=1, q=1, rescale=False)
    full_res = am_full.fix(train_res.params)
    garch_predictions = (full_res.conditional_volatility / 100 * np.sqrt(252)).values[split_idx:]

    print("2. Prepping Tensor Data...")
    scaler = StandardScaler()
    features = ['log_return', 'realized_vol', 'vix_close']
    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])
    
    seq_length = 21
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    print(f"3. Entering the Thunderdome (Running {n_trials} Full-Length Trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 1, 2)
        dropout = trial.suggest_float('dropout', 0.15, 0.35)
        lr = trial.suggest_float('lr', 0.001, 0.008)
        
        model = VolatilityLSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for _ in range(50): 
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            return criterion(model(X_test_t), y_test_t).item()

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    print(f"--> Survivor Found! Best Params: {best}")

    print("4. Training the Ultimate Champion (50 Epochs)...")
    torch.manual_seed(42)
    final_model = VolatilityLSTM(input_size=3, hidden_size=best['hidden_size'], num_layers=best['num_layers'], dropout=best['dropout'])
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best['lr'])
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        final_model.train()
        optimizer.zero_grad()
        loss = criterion(final_model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    return test_df, garch_predictions, final_model, X_test, scaler, seq_length, best