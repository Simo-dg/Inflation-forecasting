from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.utils import train_test_split_time_series
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def run_rf(df, target_col, train_start, train_end, test_start, test_end):
    X_train, y_train, _, y_test = train_test_split_time_series(df, train_start, train_end, test_start, test_end, target_col)
    X_train = X_train.dropna()
    y_train = y_train.dropna()
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    forecasts = []
    history_X = X_train.copy()
    history_y = y_train.copy()

    for t in range(len(y_test)):
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(history_X, history_y)
        X_step = df.loc[[y_test.index[t]]].drop(columns=[target_col])
        forecasts.append(model.predict(X_step)[0])
        history_X = pd.concat([history_X, X_step])
        history_y = pd.concat([history_y, y_test.iloc[t:t+1]])

    forecasts = np.array(forecasts)
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.to_numpy(), label="Observed", color="blue")
    plt.plot(y_test.index.to_numpy(), forecasts, label="Forecast", color="red", linestyle="--")
    plt.title(f"Random Forest RMSE: {rmse:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/rf_forecast.png")
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
    plt.title(f"XGBoost (RMSE={rmse:.4f})")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/xgb_forecast.png")
    plt.close()

    return forecasts, rmse


def run_mlp(df, target_col, train_start, train_end, test_start, test_end):
    X_train, y_train, _, y_test = train_test_split_time_series(df, train_start, train_end, test_start, test_end, target_col)
    X_train = X_train.dropna()
    y_train = y_train.dropna()
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(MLPRegressor(max_iter=1000, random_state=42), param_grid, cv=tscv,
                        scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_params = grid.best_params_

    forecasts = []
    history_X = X_train.copy()
    history_y = y_train.copy()

    for t in range(len(y_test)):
        history_X_scaled = scaler.fit_transform(history_X)
        model = MLPRegressor(**best_params, max_iter=1000, random_state=42)
        model.fit(history_X_scaled, history_y)

        X_step = df.loc[[y_test.index[t]]].drop(columns=[target_col])
        X_step_scaled = scaler.transform(X_step)

        forecasts.append(model.predict(X_step_scaled)[0])
        history_X = pd.concat([history_X, X_step])
        history_y = pd.concat([history_y, y_test.iloc[t:t+1]])

    forecasts = np.array(forecasts)
    rmse = np.sqrt(mean_squared_error(y_test, forecasts))

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index.to_numpy(), y_test.to_numpy(), label="Observed", color="blue")
    plt.plot(y_test.index.to_numpy(), forecasts, label="Forecast", color="red", linestyle="--")
    plt.title(f"MLP RMSE: {rmse:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/mlp_forecast.png")
    plt.close()

    return forecasts, rmse
