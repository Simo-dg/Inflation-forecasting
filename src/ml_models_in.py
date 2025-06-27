from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def run_rf_in(df, target_col):
    X = df.drop(columns=[target_col]).dropna()
    y = df[target_col].dropna()
    X, y = X.align(y, join='inner', axis=0)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    plt.figure(figsize=(10,4))
    plt.plot(y.index.to_numpy(), y.to_numpy(), label='True')
    plt.plot(y.index.to_numpy(), y_pred, label='Random Forest', linestyle='--')
    plt.title('Random Forest Forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/rf_forecast.png")
    return model, rmse

def run_xgboost_in(df, target_col):
    X = df.drop(columns=[target_col]).dropna()
    y = df[target_col].dropna()
    X, y = X.align(y, join='inner', axis=0)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    plt.figure(figsize=(10,4))
    plt.plot(y.index.to_numpy(), y.to_numpy(), label='True')
    plt.plot(y.index.to_numpy(), y_pred, label='XGBoost', linestyle='--')
    plt.title('XGBoost Forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/xgb_forecast.png")
    return model, rmse

def run_mlp_in(df, target_col):
    X = df.drop(columns=[target_col]).dropna()
    y = df[target_col].dropna()
    X, y = X.align(y, join='inner', axis=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    rmse = np.sqrt(mean_squared_error(y, y_pred))

    plt.figure(figsize=(10,4))
    plt.plot(y.index.to_numpy(), y.to_numpy(), label='True')
    plt.plot(y.index.to_numpy(), y_pred, label='MLP Forecast', linestyle='--')
    plt.title('MLP Forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/mlp_forecast.png")

    return model, rmse
