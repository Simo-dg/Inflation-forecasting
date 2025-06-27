import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def run_ols_in(df, target_col):
    X = df.drop(columns=[target_col]).dropna()
    y = df[target_col].dropna()
    X, y = X.align(y, join='inner', axis=0)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(y.index.to_numpy(), y.to_numpy(), label='True')
    plt.plot(y.index.to_numpy(), y_pred, label='OLS Prediction', linestyle='--')
    plt.title('OLS')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/ols_forecast_in.png")

    return model, rmse

def run_ols_feature_selection_in(df, target_col, n_features=5):
    X = df.drop(columns=[target_col]).dropna()
    y = df[target_col].dropna()
    X, y = X.align(y, join='inner', axis=0)

    model = LinearRegression()
    selector = RFE(model, n_features_to_select=n_features)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]

    model.fit(X[selected_features], y)
    y_pred = model.predict(X[selected_features])

    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(y.index.to_numpy(), y.to_numpy(), label='True')
    plt.plot(y.index.to_numpy(), y_pred, label='OLS + RFE Prediction', linestyle='--')
    plt.title('OLS with Feature Selection')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/ols_rfe_forecast_in.png")

    return model, selected_features, rmse
