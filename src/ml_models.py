from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.utils import train_test_split_time_series

def run_rf(df, target_col, train_start, train_end, test_start, test_end):
    _, y_train, _, y_test = train_test_split_time_series(df, train_start, train_end, test_start, test_end, target_col)
    X_train = df.loc[train_start:train_end].drop(columns=[target_col]).dropna()
    y_train = y_train.dropna()
    
    # Align train X,y
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    forecasts = []
    history_X = X_train.copy()
    history_y = y_train.copy()

    for t in range(len(y_test)):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(history_X, history_y)

        # Get test sample features for current step
        X_test_step = df.loc[[y_test.index[t]]].drop(columns=[target_col])
        
        # Predict 1-step ahead
        fcast = model.predict(X_test_step)[0]
        forecasts.append(fcast)

        # Append current test observation to history for next iteration
        history_X = pd.concat([history_X, X_test_step])
        history_y = pd.concat([history_y, y_test.iloc[t:t+1]])

    forecasts = np.array(forecasts)
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))

    # Plot only test period
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(y_test.index.to_numpy(), forecasts, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.title(f"Random Forest One-step Ahead Forecast (RMSE={rmse:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/rf_one_step_forecast_test_only.png")
    plt.close()

    return forecasts, rmse

def run_xgboost(df, target_col, train_start, train_end, test_start, test_end):
    _, y_train, _, y_test = train_test_split_time_series(df, train_start, train_end, test_start, test_end, target_col)
    X_train = df.loc[train_start:train_end].drop(columns=[target_col]).dropna()
    y_train = y_train.dropna()
    
    # Align train X,y
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    forecasts = []
    history_X = X_train.copy()
    history_y = y_train.copy()

    for t in range(len(y_test)):
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(history_X, history_y)

        X_test_step = df.loc[[y_test.index[t]]].drop(columns=[target_col])
        fcast = model.predict(X_test_step)[0]
        forecasts.append(fcast)

        history_X = pd.concat([history_X, X_test_step])
        history_y = pd.concat([history_y, y_test.iloc[t:t+1]])

    forecasts = np.array(forecasts)
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(y_test.index.to_numpy(), forecasts, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.title(f"XGBoost One-step Ahead Forecast (RMSE={rmse:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/xgb_one_step_forecast_test_only.png")
    plt.close()

    return forecasts, rmse

def run_mlp(df, target_col, train_start, train_end, test_start, test_end):
    _, y_train, _, y_test = train_test_split_time_series(df, train_start, train_end, test_start, test_end, target_col)
    X_train = df.loc[train_start:train_end].drop(columns=[target_col]).dropna()
    y_train = y_train.dropna()
    
    # Align train X,y
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    forecasts = []
    history_X = X_train.copy()
    history_y = y_train.copy()

    for t in range(len(y_test)):
        # Scale training features
        history_X_scaled = scaler.fit_transform(history_X)

        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        model.fit(history_X_scaled, history_y)

        # Scale test sample using the same scaler fitted on training
        X_test_step = df.loc[[y_test.index[t]]].drop(columns=[target_col])
        X_test_step_scaled = scaler.transform(X_test_step)

        fcast = model.predict(X_test_step_scaled)[0]
        forecasts.append(fcast)

        history_X = pd.concat([history_X, X_test_step])
        history_y = pd.concat([history_y, y_test.iloc[t:t+1]])

    forecasts = np.array(forecasts)
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.values, label="Observed (Test)", color="blue", linewidth=2)
    plt.plot(y_test.index.to_numpy(), forecasts, label="1-step Ahead Forecast Mean", linestyle="--", color="red", linewidth=2)
    plt.title(f"MLP One-step Ahead Forecast (RMSE={rmse:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/mlp_one_step_forecast_test_only.png")
    plt.close()

    return forecasts, rmse
